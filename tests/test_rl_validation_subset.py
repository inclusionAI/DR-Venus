import pandas as pd

from experiments.prepare_rl_subset import normalize_question, question_from_row
from experiments.prepare_rl_validation import build_rl_validation


def _row(index):
    return {
        "data_source": "source",
        "prompt": [{"role": "user", "content": f"question {index}"}],
        "reward_model": {"ground_truth": str(index)},
        "extra_info": {"index": index},
    }


def test_validation_builder_excludes_training_questions(tmp_path):
    source = tmp_path / "source.parquet"
    train = tmp_path / "train.parquet"
    output = tmp_path / "rl-val-2.parquet"
    pd.DataFrame([_row(index) for index in range(6)]).to_parquet(source, index=False)
    pd.DataFrame([_row(0), _row(1)]).to_parquet(train, index=False)

    manifest = build_rl_validation(
        source, train, output, total=2, seed=43, expected_source_rows=6
    )
    selected = pd.read_parquet(output).to_dict(orient="records")
    selected_questions = {
        normalize_question(question_from_row(row)) for row in selected
    }
    assert selected_questions.isdisjoint({"question 0", "question 1"})
    assert manifest["suite"] == "rl-val-2"
