// Historical observation builds are explicit. Production comparisons inspect
// memory at API boundaries; no timer, norm, histogram or counter collection.
#ifndef SHOEBOX_INLINE_DIAGNOSTICS
#if !defined(SHOEBOX_LEARNED_COMPLETION) || defined(SHOEBOX_CONTRIBUTION_CHECKS)
#define SHOEBOX_INLINE_DIAGNOSTICS 1
#else
#define SHOEBOX_INLINE_DIAGNOSTICS 0
#endif
#endif
#if SHOEBOX_INLINE_DIAGNOSTICS
#define SHOEBOX_DIAGNOSTIC(...) do { __VA_ARGS__ } while(false);
#else
#define SHOEBOX_DIAGNOSTIC(...) do {} while(false);
#endif
