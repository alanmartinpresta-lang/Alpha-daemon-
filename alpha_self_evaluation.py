
from pathlib import Path
import json, statistics

class SelfEvaluation:
    def __init__(self, history_path="ALPHA_AUTONOMOUS_HISTORY.jsonl",
                 window=8):
        self.path=Path(history_path); self.window=window

    def rows(self):
        if not self.path.exists(): return []
        out=[]
        for line in self.path.read_text(encoding="utf-8",errors="ignore").splitlines():
            try: out.append(json.loads(line))
            except: pass
        return out

    def status(self):
        rows=self.rows()[-self.window:]
        accepted=sum(r.get("status")=="accepted" for r in rows)
        rejected=sum(str(r.get("status","")).startswith("rejected") for r in rows)
        scores=[r.get("holdout") for r in rows
                if isinstance(r.get("holdout"),(int,float))]
        if len(scores)>=3:
            slope=scores[-1]-scores[0]
            mean=statistics.mean(scores)
        else:
            slope=0.0; mean=None
        stagnant=(len(rows)>=self.window and accepted==0 and
                  abs(slope)<0.0001)
        return {
            "cycles":len(rows),
            "accepted":accepted,
            "rejected":rejected,
            "holdout_mean":mean,
            "holdout_delta":slope,
            "stagnant":stagnant,
            "should_enter_improvement_phase":stagnant
        }
