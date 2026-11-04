"""Compare source-aware baseline supervision with an otherwise matched baseline.

Version-1 manifests contain original and observed run/evaluation entries per
publication policy and seed. An optional detached entry adds a separately
matched reference. The primary contrast requires detachment disabled in both
arms. Recorded observed-arm supervision counts must match independent causal
stream reconstruction, including when general source coverage is skipped.
"""
from .baseline_compare import _compare_manifest, run_cli


def compare_manifest(manifest_path, count_per_query=512, include_coverage=True):
    return _compare_manifest(manifest_path, count_per_query, include_coverage, changed_arm='observed')


def main(argv=None):
    return run_cli(compare_manifest, __doc__, argv)


if __name__ == '__main__': main()
