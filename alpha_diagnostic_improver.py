
from pathlib import Path
from dataclasses import dataclass
import ast, json, difflib

@dataclass
class Diagnosis:
    name: str
    evidence: str
    keywords: tuple

class DiagnosticImprover:
    def __init__(self, project_root):
        self.root=Path(project_root).resolve()

    def read_experiences(self, path):
        rows=[]
        if not path or not Path(path).exists(): return rows
        for line in Path(path).read_text(encoding="utf-8",errors="ignore").splitlines():
            try: rows.append(json.loads(line))
            except: pass
        return rows

    def diagnose(self, rows):
        if not rows: return []
        rewards=[float(r["reward"]) for r in rows[-1000:]
                 if isinstance(r,dict) and isinstance(r.get("reward"),(int,float))]
        actions=[str(r["action"]) for r in rows[-1000:]
                 if isinstance(r,dict) and r.get("action") is not None]
        out=[]
        if rewards and sum(x<0 for x in rewards)/len(rewards)>.35:
            out.append(Diagnosis("frequent_negative_outcomes",
                f"{sum(x<0 for x in rewards)}/{len(rewards)} recent rewards negative",
                ("reward","learn","update","action","decision")))
        if actions:
            counts={a:actions.count(a) for a in set(actions)}
            dom=max(counts,key=counts.get)
            if counts[dom]/len(actions)>.70:
                out.append(Diagnosis("action_collapse",
                    f"{counts[dom]}/{len(actions)} recent actions were '{dom}'",
                    ("action","select","policy","explore","decision")))
        return out

    def targets(self,d):
        out=[]
        for p in self.root.joinpath("aire").rglob("*.py"):
            if "__pycache__" in p.parts or p.name.startswith("alpha_"): continue
            text=p.read_text(encoding="utf-8",errors="ignore")
            try: tree=ast.parse(text)
            except: continue
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    block=ast.get_source_segment(text,n) or ""
                    score=sum(k in (n.name+" "+block).lower() for k in d.keywords)
                    if score: out.append((score,p,n))
        return sorted(out,key=lambda x:x[0],reverse=True)

    def propose(self,d,limit=3):
        out=[]
        for score,p,n in self.targets(d)[:limit]:
            text=p.read_text(encoding="utf-8")
            lines=text.splitlines(True)
            line=n.lineno-1
            indent=" "*(len(lines[line])-len(lines[line].lstrip())+4)
            marker=indent+"_alpha_diagnostic_counter = 0\n"
            newlines=lines[:line+1]+[marker]+lines[line+1:]
            out.append({
                "file":str(p.relative_to(self.root)),
                "function":n.name,"line":n.lineno,"score":score,
                "diagnosis":d.name,"evidence":d.evidence,
                "diff":"".join(difflib.unified_diff(lines,newlines))
            })
        return out
