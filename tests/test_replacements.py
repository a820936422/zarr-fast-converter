from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.resampling.replacements import (  # noqa: E402
    apply_replacement_rules,
    evaluate_expression,
    parse_replacement_rules,
)


class ReplacementRuleTests(unittest.TestCase):
    def test_multiple_literal_rules_are_first_match_and_preserve_nan(self) -> None:
        rules = parse_replacement_rules("<0, >=100", "0, 100")
        values = np.asarray([-3.0, 5.0, 100.0, 120.0, np.nan])
        actual = apply_replacement_rules(values, rules)
        np.testing.assert_equal(actual, [0.0, 5.0, 100.0, 100.0, np.nan])

    def test_statistic_expression_is_resolved_per_variable(self) -> None:
        rules = parse_replacement_rules("<=median", "mean + 1")
        self.assertEqual(rules.required_statistics, ("mean", "median"))
        actual = apply_replacement_rules(
            np.asarray([1.0, 2.0, 3.0]),
            rules,
            {"median": 2.0, "mean": 2.0},
        )
        np.testing.assert_array_equal(actual, [3.0, 3.0, 3.0])

    def test_parser_rejects_mismatched_and_unsafe_expressions(self) -> None:
        with self.assertRaisesRegex(ValueError, "数量必须一致"):
            parse_replacement_rules("<0, >1", "0")
        with self.assertRaises(ValueError):
            parse_replacement_rules("<__import__('os')", "0")
        with self.assertRaisesRegex(ValueError, "不支持的统计量"):
            evaluate_expression("unknown + 1")
        with self.assertRaisesRegex(ValueError, "幂指数"):
            evaluate_expression("2 ** 1000")

    def test_float32_replacements_do_not_double_tile_dtype(self) -> None:
        values = np.asarray([-1.0, 2.0], dtype="float32")
        actual = apply_replacement_rules(
            values,
            parse_replacement_rules("<0", "0"),
        )
        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(actual, [0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
