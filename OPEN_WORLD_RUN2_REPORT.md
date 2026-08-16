# Alpha Genesis 3 — Open World Run 2

## Protocol
- No task-specific scenario or semantic objective was injected.
- Baseline: Genesis 3 autonomy configuration.
- Candidate: only `exploration_initial` changed from 0.25 to 0.15.
- Selection rule: candidate must improve mean endogenous cumulative reward on training seeds and then be tested on held-out seeds.
- Training seeds: 20260860, 20260861.
- Hold-out seeds: 20260862, 20260863.
- Training horizon: 100 steps.
- Hold-out horizon: 300 steps.

## Results
Training mean:
- baseline: 1.04890607
- candidate: 1.05144179
- delta: +0.00253572 (+0.24%)

Hold-out mean:
- baseline: 1.2128517636
- candidate: 1.2165409655
- delta: +0.0036892019 (+0.30%)

Hold-out per seed:
- 20260862: baseline 1.2779083909; candidate 1.2732969142
- 20260863: baseline 1.1477951363; candidate 1.1597850167

Decision: ACCEPTED with a small positive mean generalization gain on the two held-out environments.

## Important correction
The previous run report described actions 7–9 as no-op. That was incorrect: the world runtime explicitly handles `push`, `grasp`, and `release` through `_material_action` in `alpha_world.py`. The current action set therefore has physical manipulation pathways for all ten named actions.

## Interpretation
The accepted change is a small, validated exploration-policy improvement, not evidence of general intelligence. It does not show that Alpha invented the parameter change itself. It shows that the change was selected by an external validation procedure and generalized slightly to held-out worlds.
