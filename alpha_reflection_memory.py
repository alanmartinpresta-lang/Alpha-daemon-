
from pathlib import Path
import json, hashlib, statistics

class ReflectionMemory:
    """Records Alpha's prediction -> outcome -> error -> consequence loop."""
    def __init__(self, path="alpha_reflection_memory.jsonl", max_records=10000):
        self.path=Path(path); self.max_records=max_records

    def signature(self, observation):
        return hashlib.sha1(repr(observation).encode()).hexdigest()[:16]

    def record(self, observation, action, predicted_value, observed_value,
               reward, strategy, step):
        e={
            "state_signature":self.signature(observation),
            "action":str(action),
            "predicted_value":float(predicted_value),
            "observed_value":float(observed_value),
            "prediction_error":float(observed_value-predicted_value),
            "reward":float(reward),
            "strategy":str(strategy),
            "step":int(step)
        }
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(e,separators=(",",":"))+"\n")
        return e

    def recent(self,n=500):
        if not self.path.exists(): return []
        rows=[]
        for line in self.path.read_text(encoding="utf-8",errors="ignore").splitlines()[-n:]:
            try: rows.append(json.loads(line))
            except: pass
        return rows

    def diagnose(self,n=500):
        rows=self.recent(n)
        if not rows: return {"records":0,"signals":[]}
        errors=[abs(float(x["prediction_error"])) for x in rows]
        rewards=[float(x["reward"]) for x in rows]
        actions=[x["action"] for x in rows]
        signals=[]
        if statistics.mean(errors)>.10:
            signals.append({"type":"prediction_instability",
                            "mean_abs_error":statistics.mean(errors)})
        if statistics.mean(rewards)<0:
            signals.append({"type":"negative_return",
                            "mean_reward":statistics.mean(rewards)})
        if actions:
            counts={a:actions.count(a) for a in set(actions)}
            dom=max(counts,key=counts.get)
            if counts[dom]/len(actions)>.70:
                signals.append({"type":"action_collapse",
                                "dominant_action":dom,
                                "share":counts[dom]/len(actions)})
        return {"records":len(rows),
                "mean_abs_prediction_error":statistics.mean(errors),
                "mean_reward":statistics.mean(rewards),
                "signals":signals}
