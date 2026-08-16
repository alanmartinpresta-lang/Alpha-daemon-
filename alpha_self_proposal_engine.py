
from pathlib import Path
import json, ast, difflib, re

class SelfProposalEngine:
    """Turns Alpha's own observed history into bounded self-edit candidates."""
    def __init__(self, project_root, history="ALPHA_REFLECTION_REAL_RUN.jsonl"):
        self.root=Path(project_root).resolve()
        self.history=self.root/history

    def history_rows(self, n=500):
        if not self.history.exists(): return []
        rows=[]
        for line in self.history.read_text(encoding="utf-8",errors="ignore").splitlines()[-n:]:
            try: rows.append(json.loads(line))
            except: pass
        return rows

    def diagnose(self):
        rows=self.history_rows()
        if not rows: return []
        rewards=[x.get("reward") for x in rows if isinstance(x.get("reward"),(int,float))]
        if not rewards: return []
        avg=sum(rewards)/len(rewards)
        signals=[]
        if avg < 0:
            signals.append({"kind":"negative_return","average_reward":avg})
        actions=[repr(x.get("action")) for x in rows if x.get("action") is not None]
        if actions:
            counts={a:actions.count(a) for a in set(actions)}
            dom=max(counts,key=counts.get)
            if counts[dom]/len(actions)>.70:
                signals.append({"kind":"action_collapse","dominant":dom,
                                 "share":counts[dom]/len(actions)})
        return signals

    def source_targets(self, signal):
        keys=("action","select","policy","decision","learn","update","reward")
        out=[]
        for p in self.root.joinpath("aire").rglob("*.py"):
            if p.name.startswith("alpha_self_") or p.name.startswith("reflection_"):
                continue
            text=p.read_text(encoding="utf-8",errors="ignore")
            try: tree=ast.parse(text)
            except: continue
            for n in ast.walk(tree):
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    block=ast.get_source_segment(text,n) or ""
                    score=sum(k in (n.name+" "+block).lower() for k in keys)
                    if score:
                        out.append((score,p,n))
        return sorted(out,key=lambda x:x[0],reverse=True)

    def propose(self, limit=3):
        signals=self.diagnose()
        proposals=[]
        for sig in signals:
            for score,p,n in self.source_targets(sig)[:limit]:
                text=p.read_text(encoding="utf-8")
                lines=text.splitlines(True)
                # Conservative real source edit: change an existing numeric
                # literal in the targeted function by +/-5%.
                block_lines=lines[n.lineno-1:n.end_lineno]
                block="".join(block_lines)
                m=re.search(r'(?<![\w.])([0-9]+(?:\.[0-9]+)?)(?![\w.])',block)
                if not m: continue
                old=float(m.group(1))
                for factor in (0.95,1.05):
                    newv=old*factor
                    replacement=repr(newv) if "." in m.group(1) else str(max(1,round(newv)))
                    newblock=block[:m.start(1)]+replacement+block[m.end(1):]
                    newlines=lines[:n.lineno-1]+[newblock]+lines[n.end_lineno:]
                    proposals.append({
                        "file":str(p.relative_to(self.root)),
                        "function":n.name,
                        "diagnosis":sig,
                        "new_source":"".join(newlines),
                        "diff":"".join(difflib.unified_diff(
                            lines,newlines,
                            fromfile=str(p.relative_to(self.root)),
                            tofile=str(p.relative_to(self.root))))
                    })
        return proposals

if __name__=="__main__":
    e=SelfProposalEngine(".")
    out=e.propose()
    Path("ALPHA_SELF_EDIT_PROPOSALS.json").write_text(json.dumps(out,indent=2))
    print(json.dumps({"diagnoses":e.diagnose(),"proposals":len(out)},indent=2))
