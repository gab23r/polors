from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polors.gurobi import GurobiPolars  # noqa: F401

from polugins import register_namespaces

register_namespaces(
    load_config=True  # Loads from pyproject.toml and polugins.toml
)
