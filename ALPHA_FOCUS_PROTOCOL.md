# Alpha Focus Protocol — Run 1

The environment is only a substrate for Alpha.

No task-specific scenario, puzzle, chase, reward hunt, or hand-authored
objective is supplied. The purpose of the environment is solely to provide
observations, actions, consequences, persistence, and enough variation for
Alpha to learn and improve its own decision process.

Primary subject of measurement:
- Alpha's internal learning;
- memory;
- prediction of consequences;
- adaptation;
- self-evaluation;
- self-directed code improvement;
- transfer/generalization.

The environment and evaluator remain external controls. They are not the
object of optimization and Alpha cannot edit them.

A candidate self-modification is accepted only if it improves Alpha's measured
performance across held-out environment seeds.
