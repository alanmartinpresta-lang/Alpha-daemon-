# Alpha — True Self-Edit Prototype

This is the first package in which Alpha's project itself is the editable
object.

The substrate remains a simple grid. It has no puzzle, chase, reward hunt,
or semantic scenario.

Inside the sandbox Alpha can inspect every Python file in its project and
create candidate source changes. Candidates are compiled in isolation and
can be adopted or rolled back.

The external evaluator and the grid are not editable.

The current prototype includes a conservative source-edit primitive. It is
not yet an unrestricted natural-language code generator. That is intentional:
the edit boundary is real, while the reasoning policy can be upgraded
independently.

Required loop:

observe Alpha diagnostics
-> inspect own project
-> formulate a hypothesis
-> generate candidate source
-> compile candidate
-> run candidate
-> compare on unseen grids
-> adopt or rollback
-> repeat
