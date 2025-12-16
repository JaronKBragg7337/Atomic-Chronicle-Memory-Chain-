# =====================================================
# ATOMIC MEMORY CHAIN v3.1 — FULL CHAIN + GAME + ROBOT + TOKENS (FIXED + STABLE)
# =====================================================
# Fixes your v3.0 issues WITHOUT narrowing your vision:
# - Worlds can be metered OR free (opposites allowed) via charge_tokens flag
# - PoW can be on/off + difficulty adjustable (prevents freezing)
# - Wallet logic fixed (no negative inserts, correct upsert)
# - Mine function fixed (typos, undefined vars, runaway loop guard)
# - Robot bridging won’t crash if robot has no tokens or no serial device
# - SQLite writes are thread-safe and resilient
#
# Run:
#   pip install pygame pyserial
#   python atomic_memory_chain_v3_1.py
# =====================================================

from __future__ import annotations

import json
import math
import os
import random
import sqlite3
import threading
import time
import uuid
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# OPTIONAL: Graceful fail if missing
try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None
    PYGAME_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except Exception:
    serial = None
    SERIAL_AVAILABLE = False


# =====================================================
# 1) TOKENIZED BLOCKCHAIN CORE
# =====================================================

@dataclass(frozen=True)
class Block:
    id: str
    timestamp: float
    previous_id: Optional[str]
    world: str
    runtime: str
    inputs_json: str
    output_state_json: str
    receipt_json: str
    bridge_witness_json: Optional[str] = None
    meta_json: Optional[str] = None
    nonce: str = "0"          # Proof-of-Work nonce (optional)
    token_cost: float = 0.0   # Cost charged for persistence (optional)

    @staticmethod
    def new(
        *,
        previous_id: Optional[str],
        world: str,
        runtime: str,
        inputs: Any,
        output_state: Any,
        receipt: Any,
        bridge_witness: Any = None,
        meta: Optional[Dict[str, Any]] = None,
        nonce: str = "0",
        token_cost: float = 0.0,
    ) -> "Block":
        return Block(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            previous_id=previous_id,
            world=world,
            runtime=runtime,
            inputs_json=json.dumps(inputs, separators=(",", ":"), sort_keys=True, default=str),
            output_state_json=json.dumps(output_state, separators=(",", ":"), sort_keys=True, default=str),
            receipt_json=json.dumps(receipt, separators=(",", ":"), sort_keys=True, default=str),
            bridge_witness_json=None if bridge_witness is None else json.dumps(
                bridge_witness, separators=(",", ":"), sort_keys=True, default=str
            ),
            meta_json=None if meta is None else json.dumps(
                meta, separators=(",", ":"), sort_keys=True, default=str
            ),
            nonce=str(nonce),
            token_cost=float(token_cost),
        )

    def inputs(self) -> Any:
        return json.loads(self.inputs_json)

    def output_state(self) -> Any:
        return json.loads(self.output_state_json)

    def receipt(self) -> Any:
        return json.loads(self.receipt_json)

    def bridge_witness(self) -> Any:
        return None if self.bridge_witness_json is None else json.loads(self.bridge_witness_json)

    def meta(self) -> Any:
        return None if self.meta_json is None else json.loads(self.meta_json)

    def pow_hash(self) -> str:
        # Hash that includes nonce so mining is real
        s = (
            f"{self.previous_id}|{self.world}|{self.runtime}|"
            f"{self.inputs_json}|{self.output_state_json}|{self.receipt_json}|{self.nonce}"
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()


class TokenizedPersistence:
    """
    SQLite-backed token ledger + block store.
    NOTE: This is "one valid realization". Worlds can ignore/replace it.
    """

    def __init__(self, db_path: str = "memory_chain.db"):
        self.db_path = db_path
        self._lock = threading.RLock()

        # Price is "tokens per byte" (keep it tiny to avoid instant bankruptcy)
        # You can tune this however you want.
        self.token_price_per_byte = 0.00001  # 0.00001 tokens per byte

        # PoW defaults (keep easy so gameplay doesn't freeze)
        self.pow_enabled_default = True
        self.pow_difficulty_prefix = "000"   # easier than "0000"
        self.pow_max_iters = 200_000         # runaway guard

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS blocks (
                    id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    previous_id TEXT,
                    world TEXT NOT NULL,
                    runtime TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    output_state_json TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    bridge_witness_json TEXT,
                    meta_json TEXT,
                    nonce TEXT,
                    token_cost REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS wallets (
                    player_id TEXT PRIMARY KEY,
                    balance REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_world_ts ON blocks(world, ts);")
            conn.commit()

    # ---------- Wallets ----------
    def get_wallet(self, player_id: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT balance FROM wallets WHERE player_id = ?",
                (player_id,),
            ).fetchone()
            return float(row[0]) if row else 0.0

    def set_wallet(self, player_id: str, balance: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO wallets(player_id, balance) VALUES(?, ?)
                ON CONFLICT(player_id) DO UPDATE SET balance = excluded.balance
                """,
                (player_id, float(balance)),
            )
            conn.commit()

    def reward_wallet(self, player_id: str, amount: float) -> None:
        amount = float(amount)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO wallets(player_id, balance) VALUES(?, ?)
                ON CONFLICT(player_id) DO UPDATE SET balance = balance + excluded.balance
                """,
                (player_id, amount),
            )
            conn.commit()

    def charge_wallet(self, player_id: str, cost: float) -> bool:
        cost = float(cost)
        if cost <= 0.0:
            return True
        with self._lock:
            bal = self.get_wallet(player_id)
            if bal < cost:
                return False
            self.set_wallet(player_id, bal - cost)
            return True

    # ---------- Blocks ----------
    def append(self, block: Block) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO blocks (
                    id, ts, previous_id, world, runtime,
                    inputs_json, output_state_json, receipt_json,
                    bridge_witness_json, meta_json, nonce, token_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block.id,
                    block.timestamp,
                    block.previous_id,
                    block.world,
                    block.runtime,
                    block.inputs_json,
                    block.output_state_json,
                    block.receipt_json,
                    block.bridge_witness_json,
                    block.meta_json,
                    block.nonce,
                    block.token_cost,
                ),
            )
            conn.commit()

    def head_id(self, world: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM blocks WHERE world=? ORDER BY ts DESC LIMIT 1",
                (world,),
            ).fetchone()
            return row[0] if row else None

    # ---------- Proof-of-Work (Optional) ----------
    def mine_nonce_for_block(self, block: Block, difficulty_prefix: Optional[str] = None) -> str:
        """
        Mines a nonce so block.pow_hash() starts with difficulty_prefix.
        Has a hard iteration cap to prevent freezing.
        """
        prefix = difficulty_prefix if difficulty_prefix is not None else self.pow_difficulty_prefix
        if not prefix:
            return "0"

        # Start with a changing base nonce attempt
        for i in range(self.pow_max_iters):
            candidate = str(i)
            # Create a new Block-like hash string by injecting nonce
            s = (
                f"{block.previous_id}|{block.world}|{block.runtime}|"
                f"{block.inputs_json}|{block.output_state_json}|{block.receipt_json}|{candidate}"
            )
            h = hashlib.sha256(s.encode("utf-8")).hexdigest()
            if h.startswith(prefix):
                return candidate

        # If not found quickly, return a fallback nonce (still valid as "optional PoW")
        return "0"


# =====================================================
# 2) WORLD BASE: Memory = Tokens (optional) + Execution
# =====================================================

class MemoryChainWorld:
    def __init__(
        self,
        name: str,
        persistence: TokenizedPersistence,
        player_id: str,
        *,
        charge_tokens: bool = True,     # opposites allowed
        pow_enabled: Optional[bool] = None,
        pow_difficulty_prefix: Optional[str] = None,
    ):
        self.name = name
        self.persistence = persistence
        self.player_id = player_id

        self.charge_tokens = bool(charge_tokens)
        self.pow_enabled = self.persistence.pow_enabled_default if pow_enabled is None else bool(pow_enabled)
        self.pow_difficulty_prefix = pow_difficulty_prefix if pow_difficulty_prefix is not None else self.persistence.pow_difficulty_prefix

        self.state: Dict[str, Any] = {}
        self._subs: List[Callable[[Block], None]] = []

    def subscribe(self, cb: Callable[[Block], None]) -> None:
        self._subs.append(cb)

    def calculate_memory_cost(self, inputs: Any) -> float:
        if not self.charge_tokens:
            return 0.0
        data_size = len(json.dumps(inputs, separators=(",", ":"), default=str).encode("utf-8"))
        return data_size * self.persistence.token_price_per_byte

    def execute_logic(self, inputs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Minimal default logic (worlds override)
        ts = time.time()
        if isinstance(inputs, dict):
            new_state = {**self.state, **inputs, "_ts": ts}
        else:
            new_state = {**self.state, "_ts": ts, "inputs": inputs}
        receipt = {"ts": ts, "world": self.name}
        return new_state, receipt

    def step(self, inputs: Any, bridge_witness: Any = None) -> Block:
        # 1) cost
        cost = self.calculate_memory_cost(inputs)

        # 2) charge (optional)
        if self.charge_tokens:
            ok = self.persistence.charge_wallet(self.player_id, cost)
            if not ok:
                raise ValueError(
                    f"Insufficient tokens. Need {cost:.6f}, have {self.persistence.get_wallet(self.player_id):.6f}"
                )

        # 3) execute
        prev_id = self.persistence.head_id(self.name)
        new_state, receipt = self.execute_logic(inputs)

        # 4) build block
        block = Block.new(
            previous_id=prev_id,
            world=self.name,
            runtime="MemoryChain",
            inputs=inputs,
            output_state=new_state,
            receipt={**receipt, "memory_cost_tokens": cost, "player": self.player_id},
            bridge_witness=bridge_witness,
            nonce="0",
            token_cost=cost,
        )

        # 5) PoW (optional)
        if self.pow_enabled and self.pow_difficulty_prefix:
            nonce = self.persistence.mine_nonce_for_block(block, self.pow_difficulty_prefix)
            block = Block(
                **{**block.__dict__, "nonce": nonce}  # type: ignore
            )

        # 6) append + update
        self.state = new_state
        self.persistence.append(block)

        # 7) notify bridges
        for cb in self._subs[:]:
            try:
                cb(block)
            except Exception:
                pass

        print(
            f"⛏️  [{self.name}] block={block.id[:8]} cost={cost:.6f} nonce={block.nonce}"
        )
        return block


# =====================================================
# 3) ASTEROIDS GAME (Tokenized)
# =====================================================

class AsteroidsGameWorld(MemoryChainWorld):
    def __init__(self, persistence: TokenizedPersistence, player_id: str = "player1"):
        super().__init__("AsteroidsGame", persistence, player_id, charge_tokens=True, pow_enabled=True, pow_difficulty_prefix="000")
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame required: pip install pygame")

        pygame.init()
        self.screen = pygame.display.set_mode((1000, 700))
        pygame.display.set_caption("Atomic Memory Chain v3.1 — Asteroids → Tokens → Robot")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.smallfont = pygame.font.Font(None, 24)

        self.ship = {"x": 500.0, "y": 350.0, "angle": 0.0, "thrust": 0.0, "bullets": []}
        self.asteroids: List[Dict[str, Any]] = []
        self.particles: List[Dict[str, Any]] = []
        self.game_score = 0
        self.spawn_asteroids(6)

    def spawn_asteroids(self, count: int = 5) -> None:
        for _ in range(count):
            self.asteroids.append(
                {
                    "x": random.randint(50, 950),
                    "y": random.randint(50, 650),
                    "dx": random.uniform(-2.0, 2.0),
                    "dy": random.uniform(-2.0, 2.0),
                    "size": random.randint(20, 45),
                }
            )

    def handle_input(self, event) -> None:
        ship = self.ship
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                ship["angle"] -= 6
            if event.key == pygame.K_RIGHT:
                ship["angle"] += 6
            if event.key == pygame.K_UP:
                ship["thrust"] = 0.35
            if event.key == pygame.K_SPACE:
                bullet = {
                    "x": ship["x"],
                    "y": ship["y"],
                    "dx": math.cos(math.radians(ship["angle"])) * 9.0,
                    "dy": -math.sin(math.radians(ship["angle"])) * 9.0,
                    "life": 55,
                }
                ship["bullets"].append(bullet)

                # SHOOT = Chronicle block (costs tokens)
                self.step(
                    {
                        "action": "shoot",
                        "player_id": self.player_id,
                        "target_x": ship["x"] + bullet["dx"] * 10,
                        "target_y": ship["y"] + bullet["dy"] * 10,
                        "score": self.game_score,
                    }
                )

    def execute_logic(self, inputs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Game-specific receipts can exist; can also be ignored (both allowed).
        ts = time.time()
        receipt = {"ts": ts, "type": "game_event"}
        new_state = dict(self.state)
        if isinstance(inputs, dict) and "action" in inputs:
            new_state["last_action"] = inputs["action"]
            new_state["score"] = inputs.get("score", self.game_score)
        new_state["_ts"] = ts
        return new_state, receipt

    def update_game(self) -> None:
        ship = self.ship

        # physics
        ship["x"] += math.cos(math.radians(ship["angle"])) * ship["thrust"]
        ship["y"] -= math.sin(math.radians(ship["angle"])) * ship["thrust"]
        ship["thrust"] *= 0.985
        ship["x"] %= 1000
        ship["y"] %= 700

        # asteroids
        for ast in self.asteroids:
            ast["x"] = (ast["x"] + ast["dx"]) % 1000
            ast["y"] = (ast["y"] + ast["dy"]) % 700

        # bullets
        ship["bullets"] = [b for b in ship["bullets"] if b["life"] > 0]
        for b in ship["bullets"]:
            b["x"] = (b["x"] + b["dx"]) % 1000
            b["y"] = (b["y"] + b["dy"]) % 700
            b["life"] -= 1

        # collisions
        remaining: List[Dict[str, Any]] = []
        for ast in self.asteroids:
            hit = False
            for b in ship["bullets"]:
                dx = abs(ast["x"] - b["x"])
                dy = abs(ast["y"] - b["y"])
                if dx < ast["size"] and dy < ast["size"]:
                    hit = True
                    self.game_score += 10
                    self.particles.extend([{"x": ast["x"], "y": ast["y"], "life": 25} for _ in range(6)])
                    break
            if not hit:
                remaining.append(ast)

        self.asteroids = remaining
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["life"] -= 1

        if not self.asteroids:
            self.spawn_asteroids(4)

        # reward example (world-defined; can be removed or inverted)
        if self.game_score and self.game_score % 200 == 0:
            self.persistence.reward_wallet(self.player_id, 1.0)

    def draw(self) -> None:
        self.screen.fill((0, 0, 0))

        # ship
        x, y, a = self.ship["x"], self.ship["y"], self.ship["angle"]
        pts = [
            (x + math.cos(math.radians(a)) * 20, y - math.sin(math.radians(a)) * 20),
            (x + math.cos(math.radians(a + 140)) * 13, y - math.sin(math.radians(a + 140)) * 13),
            (x + math.cos(math.radians(a + 220)) * 13, y - math.sin(math.radians(a + 220)) * 13),
        ]
        pygame.draw.polygon(self.screen, (0, 255, 255), [(int(px), int(py)) for px, py in pts])

        # asteroids
        for ast in self.asteroids:
            pygame.draw.circle(self.screen, (140, 140, 140), (int(ast["x"]), int(ast["y"])), ast["size"], 2)

        # bullets
        for b in self.ship["bullets"]:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(b["x"]), int(b["y"])), 3)

        # particles
        for p in self.particles:
            pygame.draw.circle(self.screen, (255, 100, 100), (int(p["x"]), int(p["y"])), 2)

        # UI
        score_text = self.font.render(f"SCORE: {self.game_score}", True, (255, 255, 255))
        tok = self.persistence.get_wallet(self.player_id)
        tokens_text = self.font.render(f"TOKENS: {tok:.2f}", True, (255, 215, 0))
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(tokens_text, (20, 60))

        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_input(event)

            self.update_game()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


# =====================================================
# 4) ROBOT WORLD (Tokenized or Free — your choice)
# =====================================================

class RobotWorld(MemoryChainWorld):
    def __init__(
        self,
        persistence: TokenizedPersistence,
        player_id: str = "robot",
        port: str = "/dev/ttyUSB0",
        *,
        charge_tokens: bool = False,   # IMPORTANT: default FREE to prevent crashes
    ):
        super().__init__("RobotWorld", persistence, player_id, charge_tokens=charge_tokens, pow_enabled=False)
        self.ser = None
        if SERIAL_AVAILABLE:
            try:
                self.ser = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                print(f"🤖 Robot serial connected: {port}")
            except Exception as e:
                print(f"🤖 Robot connect failed: {e}")
                self.ser = None

    def execute_logic(self, inputs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ts = time.time()
        receipt = {"ts": ts, "type": "robot_event"}
        new_state = dict(self.state)
        if isinstance(inputs, dict):
            new_state.update(inputs)
        new_state["_ts"] = ts

        # Real robot command (optional)
        if self.ser and isinstance(inputs, dict) and "command" in inputs:
            try:
                cmd = json.dumps(inputs["command"])
                self.ser.write((cmd + "\n").encode("utf-8"))
                resp = self.ser.readline().decode("utf-8", errors="ignore").strip()
                new_state["robot_response"] = resp
                receipt["robot_response"] = resp
                print(f"🤖 ROBOT: {resp}")
            except Exception as e:
                receipt["robot_error"] = str(e)

        return new_state, receipt


# =====================================================
# 5) CARRIER: Bridging Game → Robot
# =====================================================

class NetworkCarrier:
    def __init__(self, persistence: TokenizedPersistence):
        self.persistence = persistence
        self.worlds: Dict[str, MemoryChainWorld] = {}

    def register_world(self, world: MemoryChainWorld) -> None:
        self.worlds[world.name] = world

    def bridge_game_to_robot(self, game_block: Block) -> None:
        robot = self.worlds.get("RobotWorld")
        if robot is None:
            return

        inputs = game_block.inputs()
        if isinstance(inputs, dict) and inputs.get("action") == "shoot":
            robot.step(
                {
                    "from_game_block": game_block.id,
                    "command": {"action": "grab", "target_x": inputs.get("target_x", 500)},
                },
                bridge_witness={"game_block": game_block.id},
            )


# =====================================================
# 6) MAIN
# =====================================================

def main(robot_port: str = "/dev/ttyUSB0") -> None:
    print("🚀 ATOMIC MEMORY CHAIN v3.1")
    print("💾 Memory can be priced into tokens; worlds may also be free (opposites allowed).")
    print("🎮 SPACE = shoot (writes a block; costs tokens).")
    print("🤖 Robot receives a bridge witness and can react.")

    persistence = TokenizedPersistence("memory_chain.db")

    # Give player starting tokens
    persistence.reward_wallet("player1", 100.0)

    carrier = NetworkCarrier(persistence)

    game = AsteroidsGameWorld(persistence, "player1")
    robot = RobotWorld(persistence, "robot", robot_port, charge_tokens=False)  # FREE by default to prevent crashes

    carrier.register_world(game)
    carrier.register_world(robot)

    # Bridge: subscribe to game blocks
    def on_game_block(block: Block) -> None:
        carrier.bridge_game_to_robot(block)

    game.subscribe(on_game_block)

    print("\n🎮 Controls: LEFT/RIGHT turn, UP thrust, SPACE shoot.")
    print("💾 DB: memory_chain.db")
    print(f"💰 Start tokens (player1): {persistence.get_wallet('player1'):.2f}\n")

    print("🤖 Arduino (optional):")
    print(r'''void setup() { Serial.begin(9600); pinMode(9, OUTPUT); }
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n'); cmd.trim();
    if (cmd.indexOf("grab") >= 0) {
      digitalWrite(9, HIGH); Serial.println("ARM_GRABBED"); delay(300); digitalWrite(9, LOW);
    } else {
      Serial.println("OK");
    }
  }
}''')

    game.run()


if __name__ == "__main__":
    main("/dev/ttyUSB0")
