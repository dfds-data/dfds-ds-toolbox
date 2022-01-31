import time

import pytest

from dfds_ds_toolbox.profiling.profiling import profileit


def test_profiling_successful(tmp_path):
    @profileit(path=tmp_path, name="test")
    def dummy_function():
        time.sleep(0.1)

    dummy_function()


def test_profiling_invalid_path():
    with pytest.raises(TypeError):

        @profileit(path="invalid", name="test")
        def dummy_function():
            time.sleep(0.1)

        dummy_function()


def test_profiling_invalid_profile_name(tmp_path):
    with pytest.raises(TypeError):

        @profileit(path=tmp_path, name=2)
        def dummy_function():
            time.sleep(0.1)

        dummy_function()
