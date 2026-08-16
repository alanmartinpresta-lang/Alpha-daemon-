
from pathlib import Path
import json, hashlib

class ReflectionAdapter:
    """Records facts exposed by Genesis 3 without inventing hidden predictions."""
    def __init__(self,path="ALPHA_REFLECTION_LOG.jsonl"):
        self.path=Path(path)

    def record_step(self, report, step):
        row={
            "step":int(step),
            "action":getattr(report,"action",None),
            "reward":getattr(report,"reward",None),
            "alive":getattr(report,"alive",None),
            "state_digest":getattr(report,"state_digest",None),
            "prediction":None,
            "prediction_error":None,
            "source":"genesis3_step_report"
        }
        with self.path.open("a",encoding="utf-8") as f:
            f.write(json.dumps(row,separators=(",",":"))+"\n")
        return row
