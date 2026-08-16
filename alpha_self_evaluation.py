
from pathlib import Path
import json, statistics

class SelfEvaluation:
    def __init__(self, history_path="ALPHA_AUTONOMOUS_HISTORY.jsonl", window=8):
        self.path=Path(history_path); self.window=window
    def rows(self):
        if not self.path.exists(): return []
        rows=[]
        for line in self.path.read_text(encoding="utf-8",errors="ignore").splitlines():
            try: rows.append(json.loads(line))
            except Exception: pass
        return rows
    def status(self):
        rows=self.rows()[-self.window:]
        scores=[r.get("holdout") for r in rows
                if isinstance(r.get("holdout"),(int,float))]
        accepted=sum(r.get("status")=="accepted" for r in rows)
        delta=(scores[-1]-scores[0]) if len(scores)>=2 else 0.0
        return {
            "cycles":len(rows),
            "accepted":accepted,
            "rejected":sum(str(r.get("status","")).startswith("rejected") for r in rows),
            "holdout_mean":statistics.mean(scores) if scores else None,
            "holdout_delta":delta,
            "stagnant":len(rows)>=self.window and accepted==0 and abs(delta)<1e-12,
            "should_enter_improvement_phase":
                len(rows)>=self.window and accepted==0 and abs(delta)<1e-12
        }
