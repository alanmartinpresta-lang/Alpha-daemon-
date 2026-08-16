"""AIRE A1.8 — Alpha Genesis autonomous observer.

One Alpha, no human-defined task, no scenario, no injected objective.
The runner only advances the authoritative world and preserves the exact
individual state. The experiment observes survival, adaptation, injury,
resource use, memory, exploration and any emergent construction enabled by
the current physics/effectors.

The runner never chooses actions for Alpha and never rewards a human-selected
outcome.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from .alpha_world import AlphaWorld, create_alpha_world
from .reflection_adapter import ReflectionAdapter

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="ALPHA_GENESIS.aire")
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=20260815)
    ap.add_argument("--dt",type=float,default=1.0)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--log",default="ALPHA_GENESIS_LIFELOG.jsonl")
    args=ap.parse_args()
    if args.steps<0: raise SystemExit("--steps must be >= 0")
    path=Path(args.checkpoint)
    world=AlphaWorld.load_checkpoint(path) if args.resume else create_alpha_world(seed=args.seed,dt=args.dt)
    log=Path(args.log)
    reflection=ReflectionAdapter("ALPHA_REFLECTION_LOG.jsonl")
    start=world.time
    with log.open("a",encoding="utf-8") as f:
        for _ in range(args.steps):
            if not world.organism.alive: break
            rep=world.step(compute_digest=False)
            reflection.record_step(rep, world.time)
            f.write(json.dumps({
                "time":world.time,"alive":world.organism.alive,
                "action":rep.action,"reward":rep.reward,
                "memory":len(world.autonomy.autobiography),
                "short_memory":len(world.autonomy.memory),
                "digest":world.digest()
            },ensure_ascii=False)+"\n")
            f.flush()
    meta=world.save_checkpoint(path)
    print(json.dumps({
        "start_time":start,"end_time":world.time,
        "advanced":world.time-start,"alive":world.organism.alive,
        "autobiography":len(world.autonomy.autobiography),
        "short_memory":len(world.autonomy.memory),
        "digest":world.digest(),"checkpoint":meta["path"],
        "log":str(log)
    },indent=2))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
