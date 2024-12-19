from typing import TYPE_CHECKING, Any, TypedDict, Unpack, overload

import polars as pl

import xplor._dependencies as _dependencies

if TYPE_CHECKING:
    import gurobipy as gp
    from ortools.math_opt.python import mathopt

    from xplor.gurobi.model import XplorGurobi


class GurobiKwargs(TypedDict, total=False):
    """Optional kwargs for Gurobi."""

    deterministic: bool
    auto_update: bool


class MathOptKwargs(TypedDict, total=False):
    """Optional kwargs for MathOpt."""


class Xplor:
    """Xplor base class to wrap your OR model."""

    _backend: "XplorGurobi"

    @overload
    def __init__(self, model: "gp.Model", **backend_kwargs: Unpack[GurobiKwargs]) -> None: ...

    @overload
    def __init__(self, model: "mathopt.Model", **backend_kwargs: Unpack[MathOptKwargs]) -> None: ...

    def __init__(
        self,
        model: "gp.Model | mathopt.Model",
        **backend_kwargs: Any,
    ) -> None:
        """Initialize an Xplor instance.

        Parameters
        ----------
        model : gp.Model or math_opt.Model
            The Model to wrap
        backend_kwargs: MathOptKwargs | GurobiKwargs
            Solver specific configuration

        Notes
        -----
        The class maintains two dictionaries:
        - constrs: Stores constraints as DataFrames
        - vars: Stores variables as DataFrames

        """
        self.model = model
        self._set_backend(backend_kwargs)

        self.constrs: dict[str, pl.DataFrame] = {}
        self.vars: dict[str, pl.DataFrame] = {}

    def _set_backend(self, backend_kwargs: Any) -> None:
        if (gp_model := _dependencies.get_gurobipy_model_type()) is not None and isinstance(
            self.model, gp_model
        ):
            from xplor.gurobi.model import XplorGurobi

            self._backend = XplorGurobi(self.model, **backend_kwargs)

    def add_constrs(
        self,
        df: pl.DataFrame,
        expr: str,
        name: str | None = None,
        indices: list[str] | None = None,
    ) -> pl.DataFrame:
        """Add constraints for each row in the dataframe using a string expression.

        Parameters
        ----------
        df : pl.DataFrame
            The input DataFrame containing the data for creating constraints
        expr : str
            A string expression representing the constraint. Must include a comparison
            operator ('<=', '==', or '>='). The expression can reference column names
            and use standard mathematical operators. For example: "2*x + y <= z"
        name : str | None
            Base name for the constraints. If provided, constraints will be added as
            a new column to the DataFrame with this name.
        indices: list[str] | None
            Keys of the constraint

        Returns
        -------
        pl.DataFrame
            If name is provided, returns DataFrame with new constraints appended as a column.
            If name is None, returns the original DataFrame unchanged.

        Examples
        --------
        >>> df = pl.DataFrame({
        ...     "x": [gp.Var()],
        ...     "y": [gp.Var()],
        ...     "z": [5]
        ... })
        >>> df = df.pipe(xplor.add_constrs, model, "2*x + y <= z", name="capacity")

        Notes
        -----
        - Expression can use any column name from the DataFrame
        - Supports arithmetic operations (+, -, *, /)
        - Empty DataFrames are returned unchanged
        - The model is not automatically updated after adding constraints

        See Also
        --------
        add_vars : Function to add variables to the model

        """
        return self._backend.add_constrs(
            df=df,
            expr=expr,
            name=name,
            indices=indices,
        )

    def add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: str | None,
        *,
        lb: float | str | pl.Expr = 0.0,
        ub: float | str | pl.Expr | None = None,
        obj: float | str | pl.Expr = 0.0,
        indices: list[str] | None = None,
    ) -> pl.DataFrame:
        """Add a variable for each row in the dataframe.

        Parameters
        ----------
        df: pl.DataFrame
            The dataframe that will hold the new variables
        name : str
            The variable name
        vtype: str
            The variable type for created variables
        lb : float | str | pl.Expr
            Lower bound for created variables.
        ub : float | str | pl.Expr
            Upper bound for created variables.
        obj: float | str | pl.Expr
            Objective function coefficient for created variables.
        indices: list[str] | None
            Keys of the variables


        Returns
        -------
        DataFrame
            A new DataFrame with new Vars appended as a column

        """
        return self._backend.add_vars(
            df=df,
            name=name,
            vtype=vtype,
            lb=lb,
            ub=ub,
            obj=obj,
            indices=indices,
        )
