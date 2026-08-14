from __future__ import annotations

"""Safe, chunk-friendly value replacement rules for resampling.

The public desktop/CLI representation is two comma-separated strings. This
module normalizes that representation into immutable rules and evaluates a
small arithmetic expression language without exposing Python ``eval``.
"""

import ast
from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

import numpy as np


_CONDITION_RE = re.compile(r"^(<=|>=|==|!=|<|>)(.+)$")
_ALLOWED_FUNCTIONS = {"abs": abs, "ceil": math.ceil, "floor": math.floor}
_CONSTANTS = {"nan": float("nan"), "inf": float("inf"), "pi": math.pi, "e": math.e}
_STATISTICS = {"min", "max", "mean", "std", "median", "p50", "p95"}
_MAX_EXPRESSION_NODES = 64
_MAX_ABS_EXPONENT = 32


@dataclass(frozen=True)
class ReplacementRule:
    condition: str
    result: str
    operator: str
    threshold_expression: str
    result_expression: str


@dataclass(frozen=True)
class ReplacementRules:
    rules: tuple[ReplacementRule, ...] = ()

    @property
    def required_statistics(self) -> tuple[str, ...]:
        names: set[str] = set()
        for rule in self.rules:
            names.update(_expression_names(rule.threshold_expression))
            names.update(_expression_names(rule.result_expression))
        return tuple(sorted(names))

    @property
    def data_dependent(self) -> bool:
        return bool(self.required_statistics)

    def as_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((rule.condition, rule.result) for rule in self.rules)


def _split_items(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    items: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char in "(":
            depth += 1
        elif char in ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"替换表达式括号不匹配：{text!r}")
        elif char == "," and depth == 0:
            item = text[start:index].strip()
            if not item:
                raise ValueError("替换规则中不能包含空项。")
            items.append(item)
            start = index + 1
    if depth != 0:
        raise ValueError(f"替换表达式括号不匹配：{text!r}")
    item = text[start:].strip()
    if not item:
        raise ValueError("替换规则中不能以逗号结尾。")
    items.append(item)
    return items


def _validate_expression(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"无效数学表达式：{expression!r}") from exc
    if sum(1 for _node in ast.walk(tree)) > _MAX_EXPRESSION_NODES:
        raise ValueError(f"替换表达式过于复杂：{expression!r}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.operator, ast.unaryop)):
            continue
        if isinstance(node, (ast.UnaryOp, ast.UAdd, ast.USub)):
            continue
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
        ):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _ALLOWED_FUNCTIONS and not node.keywords and len(node.args) == 1:
                continue
        raise ValueError(
            f"表达式包含不支持的语法：{expression!r}；"
            "仅支持数字、统计量、括号、+ - * / ** % 和 abs/ceil/floor。"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"表达式只允许数值常量：{expression!r}")
    return tree


def _expression_names(expression: str) -> set[str]:
    tree = _validate_expression(expression)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id not in _ALLOWED_FUNCTIONS
        and node.id not in _CONSTANTS
    }
    unknown = names - _STATISTICS
    if unknown:
        raise ValueError("表达式包含不支持的统计量：" + ", ".join(sorted(unknown)))
    return names


def evaluate_expression(expression: str, statistics: Mapping[str, float] | None = None) -> float:
    tree = _validate_expression(expression)
    context = dict(_CONSTANTS)
    context.update({key: float(value) for key, value in (statistics or {}).items()})
    for name in _expression_names(expression):
        if name not in context:
            raise ValueError(f"表达式引用了未解析的统计量：{name}")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return float(context[node.id])
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if abs(right) > _MAX_ABS_EXPONENT:
                    raise ValueError(
                        f"替换表达式的幂指数绝对值不能超过 {_MAX_ABS_EXPONENT}。"
                    )
                return left**right
            return left % right
        if isinstance(node, ast.Call):
            return float(_ALLOWED_FUNCTIONS[node.func.id](visit(node.args[0])))
        raise TypeError(f"无法计算表达式节点：{type(node).__name__}")

    value = float(visit(tree))
    if not math.isfinite(value) and not any(name in expression.lower() for name in ("nan", "inf")):
        raise ValueError(f"表达式结果不是有限数值：{expression!r}")
    return value


def parse_replacement_rules(conditions: str, results: str) -> ReplacementRules:
    condition_items = _split_items(conditions)
    result_items = _split_items(results)
    if not condition_items and not result_items:
        return ReplacementRules()
    if len(condition_items) != len(result_items):
        raise ValueError(
            "替换条件和替换结果的数量必须一致："
            f"条件 {len(condition_items)} 项，结果 {len(result_items)} 项。"
        )
    rules: list[ReplacementRule] = []
    for condition, result in zip(condition_items, result_items):
        match = _CONDITION_RE.match(condition.replace(" ", ""))
        if match is None:
            raise ValueError(
                f"无效替换条件：{condition!r}；必须以 <、<=、>、>=、== 或 != 开头。"
            )
        operator, threshold = match.groups()
        _validate_expression(threshold)
        _validate_expression(result)
        rules.append(
            ReplacementRule(
                condition=condition,
                result=result,
                operator=operator,
                threshold_expression=threshold,
                result_expression=result,
            )
        )
    return ReplacementRules(tuple(rules))


def apply_replacement_rules(
    values: np.ndarray,
    rules: ReplacementRules,
    statistics: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Apply first-match replacement rules while preserving missing values."""

    if not rules.rules:
        return np.asarray(values)
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("替换规则只支持数值型数据变量。")
    output_dtype = (
        array.dtype
        if np.issubdtype(array.dtype, np.floating)
        else np.result_type(array.dtype, np.float64)
    )
    result = array.astype(output_dtype, copy=True)
    valid = np.isfinite(result)
    assigned = np.zeros(result.shape, dtype=bool)
    for rule in rules.rules:
        threshold = evaluate_expression(rule.threshold_expression, statistics)
        replacement = evaluate_expression(rule.result_expression, statistics)
        if rule.operator == "<":
            mask = result < threshold
        elif rule.operator == "<=":
            mask = result <= threshold
        elif rule.operator == ">":
            mask = result > threshold
        elif rule.operator == ">=":
            mask = result >= threshold
        elif rule.operator == "==":
            mask = result == threshold
        else:
            mask = result != threshold
        mask &= valid & ~assigned
        result[mask] = replacement
        assigned |= mask
    return result


def sample_statistics(
    dataset: Any,
    variable_names: tuple[str, ...],
    *,
    maximum_values: int | None = 250_000,
) -> dict[str, dict[str, float]]:
    """Collect deterministic bounded statistics for each selected variable."""

    result: dict[str, dict[str, float]] = {}
    for name in variable_names:
        variable = dataset[name]
        sizes = [int(variable.sizes[dim]) for dim in variable.dims]
        stride = 1
        while maximum_values is not None and stride < max(sizes, default=1) and (
            int(np.prod([max(1, size // stride) for size in sizes], dtype=np.int64))
            > maximum_values
        ):
            stride *= 2
        sampled = variable.isel(
            {dim: slice(None, None, stride) for dim in variable.dims}
        )
        values = np.asarray(sampled.values).reshape(-1)
        finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.number) else values
        if maximum_values is not None and finite.size > maximum_values:
            indices = np.linspace(0, finite.size - 1, maximum_values, dtype=np.int64)
            finite = finite[indices]
        if finite.size == 0:
            result[name] = {key: float("nan") for key in ("min", "max", "mean", "std", "median", "p50", "p95")}
            continue
        result[name] = {
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "median": float(np.median(finite)),
            "p50": float(np.percentile(finite, 50)),
            "p95": float(np.percentile(finite, 95)),
        }
    return result
