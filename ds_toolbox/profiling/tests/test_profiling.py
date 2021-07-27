from ds_toolbox.profiling.profiling import profileit
from pathlib import Path
import time
import pytest

def test_profiling_successful(tmp_path):
    @profileit(profiles_path=tmp_path, profile_name="test")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()

def test_profiling_invalid_path():
    with pytest.raises(TypeError):
        @profileit(profiles_path="invalid", profile_name="test")
        def dummy_function():
            time.sleep(0.1)

        dummy_function()

def test_profiling_invalid_profile_name(tmp_path):
    with pytest.raises(TypeError):
        @profileit(profiles_path=tmp_path, profile_name=2)
        def dummy_function():
            time.sleep(0.1)

        dummy_function()
    

