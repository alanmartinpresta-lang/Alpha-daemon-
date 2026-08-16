# Alpha Autonomous Daemon — Run 1

This package adds a standalone orchestration loop. Once started, it can run
Alpha, wait for Alpha-generated self-edit proposals, compile candidates,
evaluate them on fixed external train/holdout seeds, and adopt only candidates
that improve both.

The daemon does not invent improvements itself. Alpha must produce the
proposal file. This keeps the distinction between Alpha's decisions and the
external evaluator.

The evaluator seed list and sandbox boundary are external and immutable to
Alpha.
