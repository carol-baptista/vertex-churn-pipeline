"""Tests for src.inspect."""

from src.inspect import format_fairness_table


def test_format_fairness_table_columns():
    table = format_fairness_table(
        {
            "Female": {
                "n": 521,
                "recall": 0.696551724137931,
                "precision": 0.543010752688172,
                "predicted_positive_rate": 0.3570057581573896,
                "actual_churn_rate": 0.2783109404990403,
            },
            "Male": {
                "n": 536,
                "recall": 0.725925925925926,
                "precision": 0.5568181818181818,
                "predicted_positive_rate": 0.3283582089552239,
                "actual_churn_rate": 0.251865671641791,
            },
        }
    )
    assert "Group" in table
    assert "Female" in table
    assert "69.7%" in table
    assert "72.6%" in table
