import pytest

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env

from openhumsim_rl import HumanHomeostasisEnv


def test_gymnasium_checker():
    env = HumanHomeostasisEnv()
    check_env(env, skip_render_check=True)
