"""
ALPHA DAEMON — ALL IN ONE
Single-file experimental runtime.

Includes:
- changing 2-D grid world
- Alpha adaptive action loop
- prediction-error memory
- read-only Internet bridge
- self-diagnostics
- conservative self-edit proposal generation
- isolated candidate validation / rollback
- autonomous daemon loop

No claim of consciousness or general intelligence is made.
The evaluator and safety boundaries remain outside Alpha's proposed edits.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import argparse, ast, difflib, hashlib, json, random, re, statistics, subprocess, sys, time, py_compile

# ---------- WORLD ----------

@dataclass
class Cell:
    x:int
    y:int
    value:float=0.0
    blocked:bool=False

class GridWorld:
    def __init__(self,width=64,height=64,seed=0):
        self.width,self.height=width,height
        self.rng=random.Random(seed)
        self.cells=[[Cell(x,y,self.rng.random()) for y in range(height)]
                    for x in range(width)]
        self.t=0
    def observe(self,x,y,radius=2):
        out=[]
        for dx in range(-radius,radius+1):
            for dy in range(-radius,radius+1):
                xx,yy=x+dx,y+dy
                if 0<=xx<self.width and 0<=yy<self.height:
                    c=self.cells[xx][yy]
                    out.append((dx,dy,c.value,c.blocked))
        return out
    def step(self):
        for _ in range(max(1,(self.width*self.height)//80)):
            x=self.rng.randrange(self.width); y=self.rng.randrange(self.height)
            c=self.cells[x][y]
            c.value=max(0.0,min(1.0,c.value+self.rng.uniform(-0.05,0.05)))
        self.t+=1

# ---------- ALPHA CORE ----------

@dataclass
class StepReport:
    alive:bool
    action:str
    reward:float
    prediction:float
    prediction_error:float
    state_digest:str

class Alpha:
    def __init__(self,seed=0,width=64,height=64):
        self.seed=seed
        self.rng=random.Random(seed)
        self.world=GridWorld(width,height,seed)
        self.x,self.y=width//2,height//2
        self.time=0
        self.alive=True
        self.total_reward=0.0
        self.history=[]
        self.last_prediction=0.0
        self.last_action="observe"

    def digest(self):
        h=hashlib.sha256()
        h.update(f"{self.time}:{self.x}:{self.y}".encode())
        for col in self.world.cells:
            for c in col:
                h.update(f"{c.value:.7f}:{int(c.blocked)}".encode())
        return h.hexdigest()

    def observe(self):
        return self.world.observe(self.x,self.y,2)

    def predict(self,obs):
        candidates=[(v,dx,dy) for dx,dy,v,b in obs if not b]
        if not candidates: return 0.0
        return max(v for v,_,_ in candidates)

    def choose(self,obs):
        candidates=[(v,dx,dy) for dx,dy,v,b in obs if not b]
        if not candidates: return "wait",0,0
        # Exploration is kept small but nonzero.
        if self.rng.random()<0.08:
            v,dx,dy=self.rng.choice(candidates)
        else:
            v,dx,dy=max(candidates,key=lambda z:z[0])
        dx=max(-1,min(1,dx)); dy=max(-1,min(1,dy))
        return f"move_{dx}_{dy}",dx,dy

    def step(self):
        before=self.world.cells[self.x][self.y].value
        self.world.step()
        obs=self.observe()
        prediction=self.predict(obs)
        action,dx,dy=self.choose(obs)
        nx=max(0,min(self.world.width-1,self.x+dx))
        ny=max(0,min(self.world.height-1,self.y+dy))
        self.x,self.y=nx,ny
        after=self.world.cells[self.x][self.y].value
        reward=float(after-before)
        error=reward-prediction
        self.time+=1
        self.total_reward+=reward
        self.last_prediction=prediction
        self.last_action=action
        rep=StepReport(True,action,reward,prediction,error,self.digest())
        self.history.append({
            "step":self.time,"action":action,"reward":reward,
            "prediction":prediction,"prediction_error":error,
            "alive":True,"state_digest":rep.state_digest
        })
        return rep

# ---------- INTERNET: READ ONLY ----------

class AlphaInternet:
    DEFAULT_DOMAINS={
        "docs.python.org","www.wikipedia.org","en.wikipedia.org",
        "fr.wikipedia.org","arxiv.org","www.arxiv.org",
        "github.com","raw.githubusercontent.com"
    }
    def __init__(self,log="ALPHA_INTERNET_LOG.jsonl",allowed_domains=None,
                 max_bytes=1_000_000,timeout=10):
        self.log=Path(log)
        self.domains=set(allowed_domains or self.DEFAULT_DOMAINS)
        self.max_bytes=max_bytes
        self.timeout=timeout

    def get(self,url):
        u=urlparse(url)
        if u.scheme!="https" or not u.hostname:
            raise ValueError("Only HTTPS URLs are permitted.")
        host=u.hostname.lower().rstrip(".")
        if host not in self.domains:
            raise ValueError("Domain is not allowlisted: "+host)
        started=time.time()
        try:
            req=Request(u.geturl(),method="GET",
                        headers={"User-Agent":"Alpha-Research-Agent/1.0"})
            with urlopen(req,timeout=self.timeout) as r:
                data=r.read(self.max_bytes+1)
                result={"ok":True,"url":u.geturl(),
                        "status":getattr(r,"status",200),
                        "content_type":r.headers.get("Content-Type",""),
                        "body":data[:self.max_bytes].decode("utf-8","replace"),
                        "truncated":len(data)>self.max_bytes}
        except Exception as e:
            result={"ok":False,"url":u.geturl(),"error":str(e)}
        result["elapsed_s"]=round(time.time()-started,3)
        with self.log.open("a",encoding="utf-8") as f:
            f.write(json.dumps({k:v for k,v in result.items() if k!="body"},
                               separators=(",",":"))+"\n")
        return result

# ---------- SELF DIAGNOSTICS / PROPOSALS ----------

class SelfReasoning:
    def __init__(self,root="."):
        self.root=Path(root).resolve()
        self.log=self.root/"ALPHA_DAEMON_LIFE.jsonl"

    def observations(self):
        if not self.log.exists(): return []
        rows=[]
        for line in self.log.read_text(encoding="utf-8",errors="ignore").splitlines():
            try: rows.append(json.loads(line))
            except: pass
        return rows[-2000:]

    def diagnose(self):
        rows=self.observations()
        rewards=[r["reward"] for r in rows if isinstance(r.get("reward"),(int,float))]
        errors=[r["prediction_error"] for r in rows
                if isinstance(r.get("prediction_error"),(int,float))]
        signals=[]
        if rewards:
            avg=statistics.mean(rewards)
            signals.append({"kind":"mean_reward","value":avg})
            if avg<0: signals.append({"kind":"negative_return","value":avg})
        if errors:
            signals.append({"kind":"mean_prediction_error",
                             "value":statistics.mean(errors)})
        return signals

    def propose(self,limit=3):
        signals=self.diagnose()
        if not signals: return []
        # Only generate proposals from this single source file.
        path=self.root/"ALPHA_DAEMON_ALL_IN_ONE.py"
        if not path.exists(): return []
        text=path.read_text(encoding="utf-8")
        try: tree=ast.parse(text)
        except: return []
        lines=text.splitlines(True)
        proposals=[]
        for n in ast.walk(tree):
            if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
            block=ast.get_source_segment(text,n) or ""
            if not any(k in block.lower() for k in
                       ("predict","choose","explor","reward","error")):
                continue
            m=re.search(r'(?<![\w.])([0-9]+(?:\.[0-9]+)?)(?![\w.])',block)
            if not m: continue
            old=m.group(1); value=float(old)
            for factor in (0.9,1.1):
                new=repr(value*factor) if "." in old else str(max(1,round(value*factor)))
                nb=block[:m.start(1)]+new+block[m.end(1):]
                nl=lines[:n.lineno-1]+[nb]+lines[n.end_lineno:]
                proposals.append({
                    "function":n.name,"file":"ALPHA_DAEMON_ALL_IN_ONE.py",
                    "diagnosis":signals,"new_source":"".join(nl),
                    "diff":"".join(difflib.unified_diff(lines,nl))
                })
                if len(proposals)>=limit: return proposals
        return proposals

# ---------- RUNTIME ----------

def run_alpha(seed,steps,log_path):
    a=Alpha(seed=seed)
    p=Path(log_path)
    with p.open("w",encoding="utf-8") as f:
        for _ in range(steps):
            r=a.step()
            f.write(json.dumps({
                "step":a.time,"action":r.action,"reward":r.reward,
                "prediction":r.prediction,
                "prediction_error":r.prediction_error,
                "alive":r.alive,"state_digest":r.state_digest
            },separators=(",",":"))+"\n")
    return {"steps":steps,"alive":a.alive,"total_reward":a.total_reward,
            "mean_prediction_error":
                statistics.mean([x["prediction_error"] for x in a.history])}

# ---------- AUTONOMOUS DAEMON ----------

class Daemon:
    def __init__(self,root=".",cycles=10,steps=200,timeout=60):
        self.root=Path(root).resolve()
        self.cycles=cycles; self.steps=steps; self.timeout=timeout
        self.state=self.root/"ALPHA_AUTONOMOUS_STATE.json"
        self.history=self.root/"ALPHA_AUTONOMOUS_HISTORY.jsonl"

    def load_state(self):
        if self.state.exists():
            try:return json.loads(self.state.read_text())
            except:pass
        return {"cycle":0,"accepted":0,"rejected":0}

    def save(self,s):
        self.state.write_text(json.dumps(s,indent=2))

    def fingerprint(self,path=None):
        p=Path(path or __file__)
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    def candidate_test(self,source,seed):
        candidate=self.root/".alpha_candidate.py"
        candidate.write_text(source,encoding="utf-8")
        try:
            py_compile.compile(str(candidate),doraise=True)
            proc=subprocess.run(
                [sys.executable,str(candidate),"--mode","smoke",
                 "--seed",str(seed),"--steps",str(self.steps)],
                cwd=self.root,capture_output=True,text=True,
                timeout=self.timeout)
            if proc.returncode!=0:return None
            return json.loads(proc.stdout)
        except Exception:
            return None
        finally:
            if candidate.exists(): candidate.unlink()

    def log(self,row):
        with self.history.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,separators=(",",":"))+"\n")

    def run(self):
        s=self.load_state()
        for _ in range(self.cycles):
            s["cycle"]+=1
            base=run_alpha(9101,self.steps,self.root/"ALPHA_DAEMON_LIFE.jsonl")
            reasoning=SelfReasoning(self.root)
            proposals=reasoning.propose()
            self.log({"cycle":s["cycle"],"status":"diagnosis",
                      "base":base,"signals":reasoning.diagnose(),
                      "proposals":len(proposals)})
            accepted=False
            current=Path(__file__).read_text(encoding="utf-8")
            for i,proposal in enumerate(proposals):
                train=self.candidate_test(proposal["new_source"],9101)
                hold=self.candidate_test(proposal["new_source"],9201)
                if train and hold and train["total_reward"]>base["total_reward"] and \
                   hold["total_reward"]>base["total_reward"]:
                    # Do not silently overwrite the running source: save an
                    # explicitly named candidate for human inspection.
                    out=self.root/f"ALPHA_CANDIDATE_{s['cycle']}_{i}.py"
                    out.write_text(proposal["new_source"],encoding="utf-8")
                    accepted=True
                    s["accepted"]+=1
                    self.log({"cycle":s["cycle"],"status":"candidate_passed",
                              "candidate":out.name,"train":train,"holdout":hold})
                    break
                else:
                    s["rejected"]+=1
            if not accepted:
                self.log({"cycle":s["cycle"],"status":"no_candidate_adopted"})
            self.save(s)
        return s

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["smoke","daemon","research"],default="daemon")
    ap.add_argument("--seed",type=int,default=1)
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--cycles",type=int,default=10)
    ap.add_argument("--url")
    a=ap.parse_args()
    if a.mode=="smoke":
        print(json.dumps(run_alpha(a.seed,a.steps,"ALPHA_SMOKE_LIFE.jsonl"),indent=2))
    elif a.mode=="research":
        if not a.url: raise SystemExit("--url required")
        print(json.dumps(AlphaInternet().get(a.url),ensure_ascii=False,indent=2))
    else:
        print(json.dumps(Daemon(".",a.cycles,a.steps).run(),indent=2))

if __name__=="__main__":
    main()
