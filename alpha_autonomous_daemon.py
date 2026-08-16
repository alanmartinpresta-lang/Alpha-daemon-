
"""
Alpha Autonomous Daemon — Run 1

Purpose: run Alpha without an interactive human/ChatGPT loop.

External boundaries:
- evaluator/rules remain outside Alpha
- candidate versions are isolated
- rollback is mandatory on failed validation
- resource budget is explicit

This is an orchestration layer, not a claim that Alpha has human-like
consciousness or general intelligence.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, shutil, subprocess, sys, time, hashlib
from aire.alpha_self_evaluation import SelfEvaluation

class AutonomousDaemon:
    def __init__(self, root, steps=100, cycles=10, timeout=60):
        self.root=Path(root).resolve()
        self.steps=steps
        self.cycles=cycles
        self.timeout=timeout
        self.state=self.root/"ALPHA_AUTONOMOUS_STATE.json"
        self.history=self.root/"ALPHA_AUTONOMOUS_HISTORY.jsonl"

    def load_state(self):
        if self.state.exists():
            return json.loads(self.state.read_text())
        return {"cycle":0,"accepted_version":"baseline","accepted":0,"rejected":0}

    def save_state(self,s):
        self.state.write_text(json.dumps(s,indent=2))

    def fingerprint(self):
        h=hashlib.sha256()
        for p in sorted((self.root/"aire").rglob("*.py")):
            if "__pycache__" not in p.parts:
                h.update(str(p.relative_to(self.root)).encode())
                h.update(p.read_bytes())
        return h.hexdigest()[:16]

    def run_alpha(self, seed, project):
        cmd=[sys.executable,"alpha_standalone_runtime.py",
             "--steps",str(self.steps),"--seed",str(seed),
             "--log","ALPHA_DAEMON_LIFE.jsonl"]
        try:
            p=subprocess.run(cmd,cwd=project,capture_output=True,
                             text=True,timeout=self.timeout)
        except Exception as e:
            return {"ok":False,"error":str(e)}
        data=None
        try:
            data=json.loads(p.stdout)
        except Exception:
            pass
        return {"ok":p.returncode==0,"data":data,
                "stderr":p.stderr[-1000:]}
    def score(self,r):
        if not r["ok"] or not r["data"]: return None
        d=r["data"]
        # Standalone runtime exposes steps/time; reward/action details live
        # in its immutable external life log.
        return float(d.get("steps",0))/10000.0

    def log(self,row):
        with self.history.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,separators=(",",":"))+"\n")

    def run(self):
        s=self.load_state()
        evaluator=SelfEvaluation(self.root/'ALPHA_AUTONOMOUS_HISTORY.jsonl', window=8)
        # Fixed external validation seeds. Alpha cannot edit this list.
        train_seed=9101
        hold_seed=9201
        for _ in range(self.cycles):
            s["cycle"]+=1
            parent_score=self.score(self.run_alpha(train_seed,self.root))
            # Self-edit generation is intentionally delegated to Alpha's
            # project-level tools when present. No automatic patch is invented
            # by the daemon.
            evaluation=evaluator.status()
            self.log({"cycle":s["cycle"],"status":"self_evaluation",**evaluation})
            proposals=self.root/"ALPHA_SELF_EDIT_PROPOSALS.json"
            if not proposals.exists():
                self.log({"cycle":s["cycle"],"status":"waiting_for_alpha_proposal",
                          "parent_score":parent_score,
                          "version":self.fingerprint()})
                self.save_state(s)
                continue
            try:
                ps=json.loads(proposals.read_text())
            except Exception:
                ps=[]
            accepted=False
            for idx,proposal in enumerate(ps):
                candidate=Path(f"/tmp/alpha_daemon_candidate_{s['cycle']}_{idx}")
                if candidate.exists(): shutil.rmtree(candidate)
                shutil.copytree(self.root,candidate,
                                ignore=shutil.ignore_patterns("__pycache__"))
                # Proposal must contain a relative file + complete replacement.
                try:
                    target=candidate/proposal["file"]
                    target.relative_to(candidate)
                    target.write_text(proposal["new_source"],encoding="utf-8")
                    for p in (candidate/"aire").rglob("*.py"):
                        if "__pycache__" not in p.parts:
                            py_compile.compile(str(p),doraise=True)
                except Exception as e:
                    self.log({"cycle":s["cycle"],"candidate":idx,
                              "status":"rejected_compile","error":str(e)})
                    continue
                train=self.score(self.run_alpha(train_seed,candidate))
                if train is None or parent_score is None or train <= parent_score:
                    self.log({"cycle":s["cycle"],"candidate":idx,
                              "status":"rejected_train","parent":parent_score,
                              "candidate_score":train})
                    continue
                hold=self.score(self.run_alpha(hold_seed,candidate))
                if hold is None or hold <= parent_score:
                    self.log({"cycle":s["cycle"],"candidate":idx,
                              "status":"rejected_holdout","parent":parent_score,
                              "candidate_score":hold})
                    continue
                # Adopt only after held-out improvement.
                for p in candidate.rglob("*"):
                    if p.is_file():
                        rel=p.relative_to(candidate)
                        dest=self.root/rel
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(p,dest)
                s["accepted"]+=1
                accepted=True
                self.log({"cycle":s["cycle"],"candidate":idx,
                          "status":"accepted","parent":parent_score,
                          "train":train,"holdout":hold,
                          "version":self.fingerprint()})
                break
            if not accepted:
                s["rejected"]+=1
            self.save_state(s)
        return s

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cycles",type=int,default=10)
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--timeout",type=int,default=60)
    args=ap.parse_args()
    print(json.dumps(AutonomousDaemon(".",args.steps,args.cycles,args.timeout).run(),indent=2))
