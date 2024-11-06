import ast
import re

import gurobipy as gp
import polars as pl


def add_constrs_from_dataframe_args(
    df: pl.DataFrame,
    model: gp.Model,
    lhs: float | pl.Expr | pl.Series,
    sense: str,
    rhs: float | pl.Expr | pl.Series,
    name: str | None,
) -> list[gp.Constr]:
    constrs = [
        model.addLConstr(
            lhs,
            sense,
            rhs,
            name=name or "",
        )
        for lhs, rhs in df.select(lhs=lhs, rhs=rhs).rows()
    ]

    return constrs


def evaluate_comp_expr(df: pl.DataFrame, expr: str) -> tuple[pl.Series, str, pl.Series]:
    # Just get the first character of sense, to match the gurobipy enums
    lhs, rhs = re.split("[<>=]+", expr)
    sense = expr.replace(lhs, "").replace(rhs, "")[0]
    assert sense in [gp.GRB.LESS_EQUAL, gp.GRB.EQUAL, gp.GRB.GREATER_EQUAL]

    lhsseries = evaluate_expr(df, lhs.strip())
    rhsseries = evaluate_expr(df, rhs.strip())
    return lhsseries, sense, rhsseries


def evaluate_expr(df: pl.DataFrame, expr: str) -> pl.Series:
    if expr in df:
        return df[expr]
    else:
        tree = ast.parse(expr, mode="eval")
        vars = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in df.columns
        }
        if not vars:
            return df.with_columns(__polors_tmp__=eval(expr))["__polors_tmp__"]
        else:
            return pl.Series(
                [eval(expr, None, r) for r in df.select(vars).rows(named=True)],
                dtype=pl.Object,
            )
