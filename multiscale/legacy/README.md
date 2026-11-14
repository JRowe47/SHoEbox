# Original multiscale runtime

`native.cpp` and `native.hpp` are exact copies from commit
`d319c3e03f1aa77fa47919072777450940ec90aa`. Keep them unchanged.

The Python wrapper builds this separate backend for version-1 checkpoint loads
and explicit `Model(memory_profile="legacy")` comparisons. Version-1 saves
continue to use the original objectives, stream layout, initialization and
arithmetic. No new clock, address matrix or reconstruction objective is inserted
into an existing stream. Newly constructed models default to version 2.
