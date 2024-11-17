import re
from typing import Literal, overload

import gurobipy as gp
import polars as pl
from xplor import _utils


def apply_eval(self: pl.DataFrame, expr: str) -> pl.DataFrame:
    *alias, expr = re.split("=", expr)

    series = _utils.evaluate_expr(self, expr.strip())
    if alias:
        series = series.alias(alias[0].strip())

    return self.with_columns(series)


def first(expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_elements(lambda d: d[0], return_dtype=pl.Object)


def last(expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_elements(lambda d: d[-1], return_dtype=pl.Object)


def quicksum(expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_elements(gp.quicksum, return_dtype=pl.Object)


def any(expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_elements(lambda d: gp.or_(d.to_list()), return_dtype=pl.Object)


def abs(expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_elements(lambda d: gp.abs_(d.to_list()), return_dtype=pl.Object)


def read_value(expr: pl.Expr | str):
    if isinstance(expr, str):
        expr = pl.col(expr)
    return expr.map_batches(
        lambda s:
        # in case of a variable
        pl.Series([e.x for e in s])
        if s.len() and hasattr(s[0], "X")
        # in case of a linExpr
        else pl.Series([e.getValue() for e in s])
    )


def add_vars(
    self: pl.DataFrame,
    model: gp.Model,
    name: str,
    *,
    lb: float | str | pl.Expr = 0.0,
    ub: float | str | pl.Expr = gp.GRB.INFINITY,
    obj: float | str | pl.Expr = 0.0,
    indices: list[str] | None = None,
    vtype: str = gp.GRB.CONTINUOUS,
) -> pl.DataFrame:
    """Add a variable to the given model for each row in the dataframe.

    Parameters
    ----------
    model : Model
        A Gurobi model to which new variables will be added
    name : str
        Used as the appended column name, as well as the base
        name for added Gurobi variables
    lb : float | pl.Expr, optional
        Lower bound for created variables. May be a single value
        or the name of a column in the dataframe, defaults to 0.0
    ub : float | pl.Expr, optional
        Upper bound for created variables. May be a single value
        or the name of a column in the dataframe, defaults to
        :code:`GRB.INFINITY`
    obj: float | pl.Expr, optional
        Objective function coefficient for created variables.
        May be a single value, or the name of a column in the dataframe,
        defaults to 0.0
    vtype: str, optional
        Gurobi variable type for created variables, defaults
        to :code:`GRB.CONTINUOUS`

    Returns
    -------
    DataFrame
        A new DataFrame with new Vars appended as a column
    """
    lb_ = self.with_columns(lb=lb)["lb"].to_numpy() if isinstance(lb, (str, pl.Expr)) else lb
    ub_ = self.with_columns(ub=ub)["ub"].to_numpy() if isinstance(ub, (str, pl.Expr)) else ub
    obj_ = self.with_columns(obj=obj)["obj"].to_numpy() if isinstance(obj, (str, pl.Expr)) else obj
    if indices is not None:
        name_ = (
            self.select(
                pl.format(
                    "{}[{}]",
                    pl.lit(name),
                    pl.concat_str([c for c in indices], separator=","),
                )
            )
            .to_series(0)
            .to_list()
        )
    else:
        name_ = indices
    vars = model.addMVar(
        self.height,
        vtype=vtype,
        lb=lb_,
        ub=ub_,
        obj=obj_,
        name=name_,
    )
    model.update()
    return self.with_columns(pl.Series(name, vars.tolist(), dtype=pl.Object))


@overload
def add_constrs(
    df: pl.DataFrame,
    model: gp.Model,
    lhs: float | pl.Expr,
    sense: Literal["<="] | Literal["=="] | Literal[">="],
    rhs: float | pl.Expr,
    name: str | None = ...,
) -> pl.DataFrame: ...


@overload
def add_constrs(
    df: pl.DataFrame,
    model: gp.Model,
    lhs: str,
    sense: None = None,
    rhs: None = None,
    name: str | None = ...,
) -> pl.DataFrame: ...


def add_constrs(
    df: pl.DataFrame,
    model: gp.Model,
    lhs: str | float | pl.Expr,
    sense: Literal["<="] | Literal["=="] | Literal[">="] | None = None,
    rhs: float | pl.Expr | None = None,
    name: str | None = None,
) -> pl.DataFrame:
    """
    Create a constraint for each row in the dataframe.

    Can be called in 3-arg (model, data, expression) form, or 5-arg
    (model, data, lhs, sense, rhs) form.

    For 3-arg, :lhs must be a string expression including a comparison
    operator, which is evaluated over the dataframe columns to produce
    constraints.

    For 5-arg, :lhs and :rhs must refer to columns or be scalar values.
    Constraints are build as if the comparison operator were applied
    element-wise over between data[lhs] and data[rhs].
    """
    if df.height == 0:
        return df
    if isinstance(lhs, str):
        lhs_, sense_, rhs_ = _utils.evaluate_comp_expr(df, lhs)
        constrs = _utils.add_constrs_from_dataframe_args(df, model, lhs_, sense_, rhs_, name)
    else:
        assert rhs is not None
        assert sense is not None
        constrs = _utils.add_constrs_from_dataframe_args(df, model, lhs, sense, rhs, name)

    # model.update()
    if name is None:
        return df
    else:
        return df.with_columns(pl.Series(name, constrs, dtype=pl.Object))
