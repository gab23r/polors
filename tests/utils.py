from collections.abc import Generator

import gurobipy as gp
import pytest

import xplor  # noqa: F401


@pytest.fixture
def model() -> Generator[gp.Model]:
    env = gp.Env()
    model = gp.Model(env=env)
    yield model
    model.close()
    env.close()
