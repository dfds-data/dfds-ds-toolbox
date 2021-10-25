import cProfile
from functools import wraps
from pathlib import Path


def profileit(profiles_path: Path, profile_name: str):
    """Generate stats file from the execution time profile of a function.

    Used as decorator.
    Generate profiling stats using cProfile. The stats files are saved as <profiles_path>/<profile_name>.stats
    See https://docs.python.org/3/library/profile.html#module-cProfile

    Args:
        profiles_path: folder path where the stats file will be saved
        profile_name: name of the stats file
    """

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not isinstance(profiles_path, Path):
                raise TypeError(
                    f"profiles_path must be an instance of <class pathlib.Path>, got {type(profiles_path)}"
                )
            if not isinstance(profile_name, str):
                raise TypeError(
                    f"profile_name must be an instance of <class str>, got {type(profile_name)}"
                )
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            profiles_path.mkdir(exist_ok=True, parents=True)
            prof.dump_stats(profiles_path / f"{profile_name}.stats")
            return retval

        return wrapper

    return inner
