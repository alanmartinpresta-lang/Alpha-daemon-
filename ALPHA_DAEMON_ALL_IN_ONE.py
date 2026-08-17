"""
ALPHA DAEMON V1
Experimental autonomous adaptive agent.

V1 changes:
- real Q-learning memory
- state/action value estimation
- exploration decay
- learning and evaluation separated
- meaningful prediction error
- independent validation
- conservative self-improvement proposals
- candidate validation + rollback
- read-only Internet bridge

No claim of consciousness or general intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import argparse
import ast
import difflib
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
import py_compile


# ============================================================
# WORLD
# ============================================================

@dataclass
class Cell:
    x: int
    y: int
    value: float = 0.0
    blocked: bool = False


class GridWorld:
    def __init__(self, width=64, height=64, seed=0):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

        self.cells = [
            [
                Cell(x, y, self.rng.random())
                for y in range(height)
            ]
            for x in range(width)
        ]

        self.t = 0

    def observe(self, x, y, radius=2):
        result = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xx = x + dx
                yy = y + dy

                if 0 <= xx < self.width and 0 <= yy < self.height:
                    cell = self.cells[xx][yy]

                    result.append(
                        (
                            dx,
                            dy,
                            round(cell.value, 3),
                            cell.blocked,
                        )
                    )

        return result

    def step(self):
        changes = max(
            1,
            (self.width * self.height) // 80
        )

        for _ in range(changes):
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)

            cell = self.cells[x][y]

            cell.value = max(
                0.0,
                min(
                    1.0,
                    cell.value + self.rng.uniform(-0.05, 0.05)
                )
            )

        self.t += 1


# ============================================================
# REPORT
# ============================================================

@dataclass
class StepReport:
    alive: bool
    action: str
    reward: float
    prediction: float
    prediction_error: float
    state: str
    state_digest: str


# ============================================================
# ALPHA V1
# ============================================================

class Alpha:

    def __init__(
        self,
        seed=0,
        width=64,
        height=64,
        learning_rate=0.12,
        discount=0.92,
        exploration=0.20,
        exploration_min=0.02,
        exploration_decay=0.997,
        learning=True,
    ):

        self.seed = seed
        self.rng = random.Random(seed)

        self.world = GridWorld(
            width,
            height,
            seed
        )

        self.x = width // 2
        self.y = height // 2

        self.time = 0
        self.alive = True

        self.total_reward = 0.0

        # Learning parameters
        self.learning_rate = learning_rate
        self.discount = discount

        self.exploration = exploration
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        self.learning_enabled = learning

        # Q-value memory
        self.q = {}

        # Prediction statistics
        self.prediction_errors = []

        self.history = []

        self.last_state = None
        self.last_action = None
        self.last_prediction = 0.0

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    def state_from_observation(self, obs):

        values = []

        for dx, dy, value, blocked in obs:

            if blocked:
                values.append(
                    (dx, dy, -1)
                )
            else:
                # Discretisation gives Alpha a stable state space.
                bucket = min(
                    9,
                    max(
                        0,
                        int(value * 10)
                    )
                )

                values.append(
                    (dx, dy, bucket)
                )

        return tuple(values)

    def state_key(self, state):
        return repr(state)

    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    def available_actions(self):
        return [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    def action_name(self, action):
        dx, dy = action
        return f"move_{dx}_{dy}"

    # --------------------------------------------------------
    # Q MEMORY
    # --------------------------------------------------------

    def ensure_state(self, state):

        key = self.state_key(state)

        if key not in self.q:
            self.q[key] = {
                self.action_name(a): 0.0
                for a in self.available_actions()
            }

        return self.q[key]

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predict(self, state):

        values = self.ensure_state(state)

        if not values:
            return 0.0

        return max(values.values())

    # --------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------

    def choose(self, state):

        values = self.ensure_state(state)

        actions = self.available_actions()

        # Exploration
        if (
            self.learning_enabled
            and self.rng.random() < self.exploration
        ):
            return self.rng.choice(actions)

        best_value = max(values.values())

        best_actions = [
            action
            for action, value in values.items()
            if value == best_value
        ]

        chosen_name = self.rng.choice(best_actions)

        for action in actions:

            if self.action_name(action) == chosen_name:
                return action

        return (0, 0)

    # --------------------------------------------------------
    # Q UPDATE
    # --------------------------------------------------------

    def learn(
        self,
        previous_state,
        action,
        reward,
        next_state,
    ):

        if not self.learning_enabled:
            return

        previous_values = self.ensure_state(
            previous_state
        )

        next_values = self.ensure_state(
            next_state
        )

        action_name = self.action_name(action)

        old_value = previous_values[action_name]

        best_next = max(
            next_values.values()
        )

        target = reward + (
            self.discount * best_next
        )

        error = target - old_value

        previous_values[action_name] = (
            old_value
            + self.learning_rate * error
        )

        self.prediction_errors.append(
            abs(error)
        )

        # Gradually move from exploration
        # toward exploitation.
        self.exploration = max(
            self.exploration_min,
            self.exploration
            * self.exploration_decay
        )

    # --------------------------------------------------------
    # DIGEST
    # --------------------------------------------------------

    def digest(self):

        h = hashlib.sha256()

        h.update(
            f"{self.time}:{self.x}:{self.y}".encode()
        )

        for column in self.world.cells:

            for cell in column:

                h.update(
                    f"{cell.value:.7f}:"
                    f"{int(cell.blocked)}".encode()
                )

        return h.hexdigest()

    # --------------------------------------------------------
    # ONE STEP
    # --------------------------------------------------------

    def step(self):

        if not self.alive:
            raise RuntimeError(
                "Alpha is no longer alive."
            )

        previous_x = self.x
        previous_y = self.y

        # Environment changes first.
        self.world.step()

        previous_obs = self.observe(
            previous_x,
            previous_y
        )

        previous_state = self.state_from_observation(
            previous_obs
        )

        prediction = self.predict(
            previous_state
        )

        action = self.choose(
            previous_state
        )

        dx, dy = action

        nx = max(
            0,
            min(
                self.world.width - 1,
                self.x + dx
            )
        )

        ny = max(
            0,
            min(
                self.world.height - 1,
                self.y + dy
            )
        )

        before = self.world.cells[
            self.x
        ][
            self.y
        ].value

        self.x = nx
        self.y = ny

        after = self.world.cells[
            self.x
        ][
            self.y
        ].value

        # Reward is the actual environmental
        # improvement produced by the action.
        reward = float(
            after - before
        )

        next_obs = self.observe(
            self.x,
            self.y
        )

        next_state = self.state_from_observation(
            next_obs
        )

        # Prediction error is now meaningful:
        # predicted value vs actual Q target.
        next_values = self.ensure_state(
            next_state
        )

        target = reward + (
            self.discount
            * max(next_values.values())
        )

        prediction_error = (
            target - prediction
        )

        self.learn(
            previous_state,
            action,
            reward,
            next_state,
        )

        self.time += 1
        self.total_reward += reward

        self.last_state = previous_state
        self.last_action = action
        self.last_prediction = prediction

        report = StepReport(
            alive=True,
            action=self.action_name(action),
            reward=reward,
            prediction=prediction,
            prediction_error=prediction_error,
            state=self.state_key(previous_state),
            state_digest=self.digest(),
        )

        self.history.append(
            {
                "step": self.time,
                "action": report.action,
                "reward": report.reward,
                "prediction": report.prediction,
                "prediction_error": report.prediction_error,
                "alive": True,
                "exploration": self.exploration,
                "q_states": len(self.q),
                "state_digest": report.state_digest,
            }
        )

        return report

    def observe(self, x=None, y=None):

        if x is None:
            x = self.x

        if y is None:
            y = self.y

        return self.world.observe(
            x,
            y,
            radius=2
        )


# ============================================================
# INTERNET — READ ONLY
# ============================================================

class AlphaInternet:

    DEFAULT_DOMAINS = {
        "docs.python.org",
        "www.wikipedia.org",
        "en.wikipedia.org",
        "fr.wikipedia.org",
        "arxiv.org",
        "www.arxiv.org",
        "github.com",
        "raw.githubusercontent.com",
    }

    def __init__(
        self,
        log="ALPHA_INTERNET_LOG.jsonl",
        allowed_domains=None,
        max_bytes=1_000_000,
        timeout=10,
    ):

        self.log = Path(log)

        self.domains = set(
            allowed_domains
            or self.DEFAULT_DOMAINS
        )

        self.max_bytes = max_bytes
        self.timeout = timeout

    def get(self, url):

        parsed = urlparse(url)

        if (
            parsed.scheme != "https"
            or not parsed.hostname
        ):
            raise ValueError(
                "Only HTTPS URLs are permitted."
            )

        host = parsed.hostname.lower().rstrip(".")

        if host not in self.domains:
            raise ValueError(
                "Domain is not allowlisted: "
                + host
            )

        started = time.time()

        try:

            request = Request(
                parsed.geturl(),
                method="GET",
                headers={
                    "User-Agent":
                    "Alpha-Research-Agent/1.0"
                },
            )

            with urlopen(
                request,
                timeout=self.timeout
            ) as response:

                data = response.read(
                    self.max_bytes + 1
                )

                result = {
                    "ok": True,
                    "url": parsed.geturl(),
                    "status": getattr(
                        response,
                        "status",
                        200
                    ),
                    "content_type":
                        response.headers.get(
                            "Content-Type",
                            ""
                        ),
                    "body":
                        data[
                            :self.max_bytes
                        ].decode(
                            "utf-8",
                            "replace"
                        ),
                    "truncated":
                        len(data)
                        > self.max_bytes,
                }

        except Exception as exc:

            result = {
                "ok": False,
                "url": parsed.geturl(),
                "error": str(exc),
            }

        result["elapsed_s"] = round(
            time.time() - started,
            3
        )

        with self.log.open(
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    {
                        key: value
                        for key, value
                        in result.items()
                        if key != "body"
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

        return result


# ============================================================
# RUNTIME
# ============================================================

def run_alpha(
    seed,
    steps,
    log_path,
    learning=True,
):

    alpha = Alpha(
        seed=seed,
        learning=learning,
    )

    path = Path(log_path)

    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for _ in range(steps):

            report = alpha.step()

            file.write(
                json.dumps(
                    {
                        "step":
                            alpha.time,
                        "action":
                            report.action,
                        "reward":
                            report.reward,
                        "prediction":
                            report.prediction,
                        "prediction_error":
                            report.prediction_error,
                        "alive":
                            report.alive,
                        "exploration":
                            alpha.exploration,
                        "q_states":
                            len(alpha.q),
                        "state_digest":
                            report.state_digest,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    mean_error = (
        statistics.mean(
            alpha.prediction_errors
        )
        if alpha.prediction_errors
        else 0.0
    )

    return {
        "steps": steps,
        "alive": alpha.alive,
        "total_reward":
            alpha.total_reward,
        "mean_absolute_learning_error":
            mean_error,
        "q_states":
            len(alpha.q),
        "final_exploration":
            alpha.exploration,
        "learning_enabled":
            learning,
    }


# ============================================================
# LEARNING / EVALUATION EXPERIMENT
# ============================================================

def run_experiment(
    seed=9101,
    train_steps=1000,
    evaluation_steps=1000,
):

    train = run_alpha(
        seed,
        train_steps,
        "ALPHA_TRAINING.jsonl",
        learning=True,
    )

    evaluation = run_alpha(
        seed + 1,
        evaluation_steps,
        "ALPHA_EVALUATION.jsonl",
        learning=False,
    )

    return {
        "training": train,
        "evaluation": evaluation,
    }


# ============================================================
# SELF DIAGNOSTICS
# ============================================================

class SelfReasoning:

    def __init__(self, root="."):

        self.root = Path(
            root
        ).resolve()

        self.log = (
            self.root
            / "ALPHA_DAEMON_LIFE.jsonl"
        )

    def observations(self):

        if not self.log.exists():
            return []

        rows = []

        for line in self.log.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines():

            try:
                rows.append(
                    json.loads(line)
                )

            except Exception:
                pass

        return rows[-2000:]

    def diagnose(self):

        rows = self.observations()

        rewards = [
            row["reward"]
            for row in rows
            if isinstance(
                row.get("reward"),
                (int, float)
            )
        ]

        errors = [
            abs(row["prediction_error"])
            for row in rows
            if isinstance(
                row.get("prediction_error"),
                (int, float)
            )
        ]

        signals = []

        if rewards:

            signals.append(
                {
                    "kind":
                        "mean_reward",
                    "value":
                        statistics.mean(
                            rewards
                        ),
                }
            )

        if errors:

            signals.append(
                {
                    "kind":
                        "mean_absolute_error",
                    "value":
                        statistics.mean(
                            errors
                        ),
                }
            )

        return signals

    def propose(self, limit=3):

        signals = self.diagnose()

        if not signals:
            return []

        source_path = (
            self.root
            / "ALPHA_DAEMON_ALL_IN_ONE.py"
        )

        if not source_path.exists():
            return []

        source = source_path.read_text(
            encoding="utf-8"
        )

        try:
            tree = ast.parse(source)

        except Exception:
            return []

        lines = source.splitlines(
            keepends=True
        )

        proposals = []

        # Only touch explicit learning
        # parameters.
        allowed_names = {
            "learning_rate",
            "discount",
            "exploration",
            "exploration_decay",
        }

        for node in ast.walk(tree):

            if not isinstance(
                node,
                ast.Assign
            ):
                continue

            if len(node.targets) != 1:
                continue

            target = node.targets[0]

            if not isinstance(
                target,
                ast.Name
            ):
                continue

            if target.id not in allowed_names:
                continue

            if not isinstance(
                node.value,
                (ast.Constant,)
            ):
                continue

            if not isinstance(
                node.value.value,
                (int, float)
            ):
                continue

            old = float(
                node.value.value
            )

            candidates = [
                old * 0.9,
                old * 1.1,
            ]

            for new_value in candidates:

                if (
                    target.id
                    == "learning_rate"
                ):
                    new_value = max(
                        0.001,
                        min(
                            1.0,
                            new_value
                        )
                    )

                elif (
                    target.id
                    == "discount"
                ):
                    new_value = max(
                        0.0,
                        min(
                            0.999,
                            new_value
                        )
                    )

                elif (
                    target.id
                    == "exploration"
                ):
                    new_value = max(
                        0.01,
                        min(
                            1.0,
                            new_value
                        )
                    )

                elif (
                    target.id
                    == "exploration_decay"
                ):
                    new_value = max(
                        0.9,
                        min(
                            0.99999,
                            new_value
                        )
                    )

                new_source = source

                old_literal = repr(
                    node.value.value
                )

                new_literal = repr(
                    new_value
                )

                new_source = (
                    new_source.replace(
                        old_literal,
                        new_literal,
                        1,
                    )
                )

                proposals.append(
                    {
                        "parameter":
                            target.id,
                        "old":
                            old,
                        "new":
                            new_value,
                        "diagnosis":
                            signals,
                        "source":
                            new_source,
                    }
                )

                if len(proposals) >= limit:
                    return proposals

        return proposals


# ============================================================
# AUTONOMOUS DAEMON
# ============================================================

class Daemon:

    def __init__(
        self,
        root=".",
        cycles=1,
        steps=500,
        timeout=60,
    ):

        self.root = Path(
            root
        ).resolve()

        self.cycles = cycles
        self.steps = steps
        self.timeout = timeout

        self.state = (
            self.root
            / "ALPHA_AUTONOMOUS_STATE.json"
        )

        self.history = (
            self.root
            / "ALPHA_AUTONOMOUS_HISTORY.jsonl"
        )

    def load_state(self):

        if self.state.exists():

            try:
                return json.loads(
                    self.state.read_text()
                )

            except Exception:
                pass

        return {
            "cycle": 0,
            "accepted": 0,
            "rejected": 0,
        }

    def save_state(self, state):

        self.state.write_text(
            json.dumps(
                state,
                indent=2
            ),
            encoding="utf-8",
        )

    def log(self, row):

        with self.history.open(
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def candidate_test(
        self,
        source,
        seed,
        learning=True,
    ):

        candidate = (
            self.root
            / ".alpha_candidate.py"
        )

        candidate.write_text(
            source,
            encoding="utf-8"
        )

        try:

            py_compile.compile(
                str(candidate),
                doraise=True,
            )

            process = subprocess.run(
                [
                    sys.executable,
                    str(candidate),
                    "--mode",
                    "smoke",
                    "--seed",
                    str(seed),
                    "--steps",
                    str(self.steps),
                    "--learning",
                    "1" if learning else "0",
                ],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if process.returncode != 0:
                return None

            return json.loads(
                process.stdout
            )

        except Exception:
            return None

        finally:

            if candidate.exists():
                candidate.unlink()

    def run(self):

        state = self.load_state()

        for _ in range(self.cycles):

            state["cycle"] += 1

            # Baseline.
            baseline = run_alpha(
                9101,
                self.steps,
                self.root
                / "ALPHA_DAEMON_LIFE.jsonl",
                learning=True,
            )

            reasoning = SelfReasoning(
                self.root
            )

            proposals = reasoning.propose()

            self.log(
                {
                    "cycle":
                        state["cycle"],
                    "status":
                        "diagnosis",
                    "baseline":
                        baseline,
                    "signals":
                        reasoning.diagnose(),
                    "proposals":
                        len(proposals),
                }
            )

            accepted = False

            source_path = (
                self.root
                / "ALPHA_DAEMON_ALL_IN_ONE.py"
            )

            current_source = (
                source_path.read_text(
                    encoding="utf-8"
                )
            )

            for index, proposal in enumerate(
                proposals
            ):

                train = self.candidate_test(
                    proposal["source"],
                    9101,
                    True,
                )

                holdout = self.candidate_test(
                    proposal["source"],
                    9201,
                    False,
                )

                if (
                    train
                    and holdout
                    and train["total_reward"]
                    > baseline["total_reward"]
                    and holdout["total_reward"]
                    > baseline["total_reward"]
                ):

                    output = (
                        self.root
                        / (
                            f"ALPHA_CANDIDATE_"
                            f"{state['cycle']}_"
                            f"{index}.py"
                        )
                    )

                    output.write_text(
                        proposal["source"],
                        encoding="utf-8",
                    )

                    state["accepted"] += 1
                    accepted = True

                    self.log(
                        {
                            "cycle":
                                state["cycle"],
                            "status":
                                "candidate_passed",
                            "candidate":
                                output.name,
                            "parameter":
                                proposal[
                                    "parameter"
                                ],
                            "old":
                                proposal["old"],
                            "new":
                                proposal["new"],
                            "baseline":
                                baseline,
                            "train":
                                train,
                            "holdout":
                                holdout,
                        }
                    )

                    break

                else:

                    state["rejected"] += 1

                    self.log(
                        {
                            "cycle":
                                state["cycle"],
                            "status":
                                "candidate_rejected",
                            "parameter":
                                proposal[
                                    "parameter"
                                ],
                            "old":
                                proposal["old"],
                            "new":
                                proposal["new"],
                            "train":
                                train,
                            "holdout":
                                holdout,
                        }
                    )

            if not accepted:

                self.log(
                    {
                        "cycle":
                            state["cycle"],
                        "status":
                            "no_candidate_adopted",
                    }
                )

            self.save_state(state)

        return state


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "experiment",
            "daemon",
            "research",
        ],
        default="smoke",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--url"
    )

    parser.add_argument(
        "--learning",
        choices=["0", "1"],
        default="1",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # SMOKE
    # --------------------------------------------------------

    if args.mode == "smoke":

        result = run_alpha(
            args.seed,
            args.steps,
            "ALPHA_SMOKE_LIFE.jsonl",
            learning=args.learning == "1",
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        return

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    if args.mode == "experiment":

        result = run_experiment(
            seed=args.seed,
            train_steps=args.steps,
            evaluation_steps=args.steps,
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        return

    # --------------------------------------------------------
    # INTERNET
    # --------------------------------------------------------

    if args.mode == "research":

        if not args.url:
            raise SystemExit(
                "--url required"
            )

        result = AlphaInternet().get(
            args.url
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    # --------------------------------------------------------
    # DAEMON
    # --------------------------------------------------------

    daemon = Daemon(
        root=".",
        cycles=args.cycles,
        steps=args.steps,
    )

    result = daemon.run()

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
