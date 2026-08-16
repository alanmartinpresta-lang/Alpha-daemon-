"""Minimal persistent Alpha runner.

Usage:
  python -m aire.run_alpha --steps 3600 --checkpoint alpha.aire
  python -m aire.run_alpha --resume --steps 3600 --checkpoint alpha.aire

The runner never supplies goals or a behavior scenario. It only advances the
autonomous Alpha World and persists the exact individual state.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from .alpha_world import AlphaWorld, create_alpha_world

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="alpha_checkpoint.aire")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2011)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.steps < 0:
        raise SystemExit("--steps must be >= 0")
    path = Path(args.checkpoint)
    if args.resume:
        world = AlphaWorld.load_checkpoint(path)
    else:
        world = create_alpha_world(seed=args.seed, dt=args.dt)
    for _ in range(args.steps):
        if not world.organism.alive:
            break
        world.step(compute_digest=False)
    meta = world.save_checkpoint(path)
    print(f"time={world.time:.9f}")
    print(f"steps={world.autonomy.steps}")
    print(f"alive={world.organism.alive}")
    print(f"memory={len(world.autonomy.memory)}")
    print(f"digest={world.digest()}")
    print(f"checkpoint={meta['path']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
