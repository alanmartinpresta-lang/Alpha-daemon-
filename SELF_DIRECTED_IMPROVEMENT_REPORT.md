# Self-directed improvement — Run 2

Alpha observed its own reward diagnostics and generated a bounded set of
possible internal changes. The candidates were evaluated on training seeds,
then the winner was tested on held-out seeds.

**Selected change:** learning_rate 0.16 → 0.12

Training mean:
- baseline: 1.11252275
- selected: 1.12245657

Held-out mean:
- baseline: 1.18894102
- selected: 1.19949250
- relative improvement: 0.887%

The selected version is included in this archive.

This is evidence of a validated code change selected by a bounded
self-directed controller. It is not evidence of consciousness or general
intelligence, and the controller is not an unrestricted language model.
