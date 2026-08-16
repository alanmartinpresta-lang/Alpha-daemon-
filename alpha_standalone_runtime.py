
"""Standalone Alpha runtime supervisor.

Runs Alpha's real Genesis world directly, without spreadsheet/artifact tooling.
The world and evaluator stay inside the Alpha experiment package; no external
ChatGPT/notebook hooks are imported.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, random, time, traceback

from aire.alpha_world import create_alpha_world

def run(seed, steps, log_path):
    world=create_alpha_world(seed=seed)
    log=Path(log_path)
    with log.open("w",encoding="utf-8") as f:
        for i in range(steps):
            rep=world.step(compute_digest=False)
            row={
                "step":i+1,
                "action":getattr(rep,"action",None),
                "reward":getattr(rep,"reward",None),
                "alive":getattr(rep,"alive",None),
                "state_digest":getattr(rep,"state_digest",None),
                "world_time":getattr(world,"time",None)
            }
            f.write(json.dumps(row,separators=(",",":"))+"\n")
            f.flush()
            if not getattr(rep,"alive",True):
                break
    return {
        "steps":i+1,
        "alive":getattr(world,"alive",None),
        "time":getattr(world,"time",None),
        "log":str(log)
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--log",default="ALPHA_STANDALONE_LIFE.jsonl")
    a=ap.parse_args()
    print(json.dumps(run(a.seed,a.steps,a.log),indent=2))
