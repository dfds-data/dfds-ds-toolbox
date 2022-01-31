import cProfile
from functools import wraps
from pathlib import Path
from typing import Callable


def profileit(path: Path, name: str) -> Callable:
    """Generate stats file from the execution time profile of a function.

    Used as decorator.
    Generate profiling stats using cProfile. The stats files are saved as <path>/<name>.stats
    See https://docs.python.org/3/library/profile.html#module-cProfile

    Args:
        path: folder path where the stats file will be saved
        name: name of the stats file
    """

    def _inner(func: Callable) -> Callable:
        @wraps(func)
        def _wrapper(*args, **kwargs):
            if not isinstance(path, Path):
                raise TypeError(
                    f"path must be an instance of <class pathlib.Path>, got {type(path)}"
                )
            if not isinstance(name, str):
                raise TypeError(f"name must be an instance of <class str>, got {type(name)}")
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            path.mkdir(exist_ok=True, parents=True)
            prof.dump_stats(path / f"{name}.stats")
            return retval

        return _wrapper

    return _inner
