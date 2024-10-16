import gurobipy as gp
import polors  # noqa: F401
import pytest


@pytest.fixture
def model():
    env = gp.Env()
    model = gp.Model(env=env)
    yield model
    model.close()
    env.close()
