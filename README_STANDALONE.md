# AIRE Genesis 3 — standalone continuation

This package contains the complete A1.10 engine (`aire/`) and a working Genesis 3
checkpoint.

Validated before packaging:
- Python compilation: all bundled .py files pass.
- Fresh Genesis 3 block: 0 -> 3000 s completed.
- Alpha alive: yes.
- Deaths: 0.
- Autobiographical memory: 3000.
- 8 material objects persisted.
- Core temperature: 310.174–312.430 K.
- Checkpoint save/load round-trip: exact match at 3000 s.
- `resume_genesis3.py` resumes the same Alpha and runs the next exact 3000-s block.

Run:
    python resume_genesis3.py

Each execution advances exactly one 3000-s block and updates GENESIS3.aire,
GENESIS3_STATE.json and GENESIS3_BLOCKS.jsonl.

No semantic construction goal, crafting recipe, or construction reward is added.
