
"""Generate conservative, auditable self-edit proposals from observations."""
from pathlib import Path
import ast, json, re, difflib, statistics

class SelfReasoning:
    def __init__(self, project_root="."):
        self.root=Path(project_root).resolve()

    def observations(self):
        p=self.root/"ALPHA_DAEMON_LIFE.jsonl"
        if not p.exists(): return []
        out=[]
        for line in p.read_text(encoding="utf-8",errors="ignore").splitlines():
            try: out.append(json.loads(line))
            except Exception: pass
        return out[-2000:]

    def diagnose(self):
        rows=self.observations()
        rewards=[r.get("reward") for r in rows if isinstance(r.get("reward"),(int,float))]
        signals=[]
        if rewards:
            avg=statistics.mean(rewards)
            if avg < 0: signals.append({"kind":"negative_return","value":avg})
        return signals

    def propose(self, limit=3):
        signals=self.diagnose()
        if not signals: return []
        proposals=[]
        for p in sorted(self.root.rglob("*.py")):
            if "alpha_self_" in p.name or "__pycache__" in p.parts: continue
            text=p.read_text(encoding="utf-8",errors="ignore")
            try: tree=ast.parse(text)
            except Exception: continue
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    block=ast.get_source_segment(text,n) or ""
                    m=re.search(r'(?<![\w.])([0-9]+(?:\.[0-9]+)?)(?![\w.])',block)
                    if not m: continue
                    old=m.group(1); value=float(old)
                    lines=text.splitlines(True)
                    for factor in (0.9,1.1):
                        new=repr(value*factor) if "." in old else str(max(1,round(value*factor)))
                        nb=block[:m.start(1)]+new+block[m.end(1):]
                        nl=lines[:n.lineno-1]+[nb]+lines[n.end_lineno:]
                        proposals.append({
                            "file":str(p.relative_to(self.root)),
                            "function":n.name,
                            "diagnosis":signals,
                            "new_source":"".join(nl),
                            "diff":"".join(difflib.unified_diff(lines,nl))
                        })
                        if len(proposals)>=limit: return proposals
        return proposals

if __name__=="__main__":
    p=SelfReasoning(".").propose()
    Path("ALPHA_SELF_EDIT_PROPOSALS.json").write_text(json.dumps(p,indent=2))
    print(json.dumps({"proposals":len(p)},indent=2))
