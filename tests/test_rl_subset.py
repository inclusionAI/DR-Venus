from copy import deepcopy

import pytest

from experiments.prepare_rl_subset import normalize_question, select_training_questions


def row(source, index, question):
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": question}],
        "ability": "research",
        "reward_model": {"ground_truth": "answer"},
        "extra_info": {"index": index},
    }


def test_selection_is_deterministic_and_input_order_independent():
    rows = [row("a", f"a{i}", f"Question A {i}") for i in range(8)]
    rows += [row("b", f"b{i}", f"Question B {i}") for i in range(4)]

    first, first_meta = select_training_questions(rows, total=6, seed=42)
    second, second_meta = select_training_questions(list(reversed(rows)), total=6, seed=42)

    ids = lambda records: [record["extra_info"]["rl_question_id"] for record in records]
    assert ids(first) == ids(second)
    assert first_meta == second_meta
    assert first_meta["selected_counts"] == {"a": 4, "b": 2}


def test_duplicate_normalized_questions_are_selected_at_most_once():
    rows = [
        row("a", "z", "  Same   QUESTION "),
        row("a", "a", "same question"),
        row("a", "other", "Other question"),
    ]

    selected, metadata = select_training_questions(rows, total=2, seed=1)

    assert len(selected) == 2
    assert metadata["duplicate_rows_removed"] == 1
    assert len(set(metadata["question_ids"])) == 2
    duplicate = next(item for item in selected if normalize_question(item["prompt"][0]["content"]) == "same question")
    assert duplicate["extra_info"]["index"] == "a"


def test_small_source_gets_representation_when_budget_allows():
    rows = [row("large", f"l{i}", f"Large {i}") for i in range(20)]
    rows += [row("small", "s0", "Small zero")]

    _, metadata = select_training_questions(rows, total=4, seed=5)

    assert metadata["selected_counts"]["small"] == 1
    assert sum(metadata["selected_counts"].values()) == 4


def test_cross_source_duplicate_is_deduplicated_globally_and_reported():
    rows = [
        row("b", "b1", "Shared question"),
        row("a", "a1", "shared  question"),
        row("a", "a2", "Unique question"),
    ]

    selected, metadata = select_training_questions(rows, total=2, seed=42)

    assert metadata["cross_source_duplicate_questions"] == 1
    assert metadata["duplicate_rows_removed"] == 1
    assert len(selected) == 2


def test_bad_rows_and_impossible_size_fail_closed():
    with pytest.raises(ValueError, match="user question"):
        select_training_questions([row("a", "1", "")], total=1)
    with pytest.raises(ValueError, match="total"):
        select_training_questions([row("a", "1", "one")], total=2)


def test_selected_rows_are_copies():
    rows = [row("a", "1", "one")]
    original = deepcopy(rows)

    selected, _ = select_training_questions(rows, total=1)
    selected[0]["extra_info"]["new"] = True

    assert rows == original
