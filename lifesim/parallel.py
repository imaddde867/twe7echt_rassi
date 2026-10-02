"""Small helper shared by the tools that fan work out over processes."""
import os


def default_workers() -> int:
    """One worker per core this process may use (respects a Slurm allocation)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1
