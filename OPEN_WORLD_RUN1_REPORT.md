# Alpha Genesis 3 — Open World Run 1

This run used the latest validated self-improving branch as the starting code.
No task-specific scenario was supplied. Alpha received only its endogenous
homeostatic viability signal and its existing actuator affordances.

Run:
- seed: 20260850
- steps: 3000
- alive at end: true
- cumulative endogenous reward: 3.630220009113996
- distinct learned sensory states: 15
- autobiographical experiences: 3000
- short memory: 512
- final epsilon: 0.02

Observed action counts:
- action 0: 546
- action 1: 29
- action 2: 32
- action 3: 37
- action 4: 32
- action 5: 29
- action 6: 413
- action 7: 733
- action 8: 532
- action 9: 617

Important limitation:
The current experimental branch exposes 10 action labels, while the
action-command function only maps the first 7 to physical motor effects.
Actions 7-9 therefore currently behave as no-op commands. This is a code
limitation discovered during this run and should be fixed before interpreting
those action frequencies as meaningful physical behaviours.

No new self-modification was accepted during this open-world run. Therefore
the code itself has not been silently altered. The package contains the exact
tested code plus the learned checkpoint and run report.
