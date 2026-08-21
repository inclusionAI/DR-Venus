import json

import pytest

from evaluation.build_eval_core import (
    SELECTION_ALGORITHM,
    allocate_stratified_counts,
    build_eval_core,
    select_stratified_rows,
)


def make_row(source, index):
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": f"question {source} {index}"}],
        "ability": "search & question answering",
        "reward_model": {"ground_truth": f"answer {index}", "style": "unknown"},
        "extra_info": {"index": f"{source}_{index}", "split": "dev"},
    }


def test_actual_distribution_allocates_six_by_sixteen_plus_four():
    counts = {
        "2wiki": 512,
        "popqa": 512,
        "tq": 512,
        "hotpotqa": 512,
        "nq": 512,
        "musique": 512,
        "Bamboogle": 125,
    }
    allocation = allocate_stratified_counts(counts, 100)
    assert allocation["Bamboogle"] == 4
    assert all(allocation[source] == 16 for source in counts if source != "Bamboogle")
    assert sum(allocation.values()) == 100


def test_selection_is_independent_of_input_row_order():
    rows = [make_row(source, index) for source in ("a", "b") for index in range(10)]
    selected_a, _, counts_a = select_stratified_rows(rows, total=6, seed=7)
    selected_b, _, counts_b = select_stratified_rows(reversed(rows), total=6, seed=7)
    ids_a = [row["extra_info"]["eval_sample_id"] for row in selected_a]
    ids_b = [row["extra_info"]["eval_sample_id"] for row in selected_b]
    assert ids_a == ids_b
    assert counts_a == counts_b == {"a": 3, "b": 3}
    assert all(row["extra_info"]["eval_selection_seed"] == 7 for row in selected_a)


def test_selection_changes_with_seed():
    rows = [make_row("source", index) for index in range(30)]
    selected_a, _, _ = select_stratified_rows(rows, total=5, seed=1)
    selected_b, _, _ = select_stratified_rows(rows, total=5, seed=2)
    ids_a = {row["extra_info"]["eval_sample_id"] for row in selected_a}
    ids_b = {row["extra_info"]["eval_sample_id"] for row in selected_b}
    assert ids_a != ids_b


def test_duplicate_dataset_id_is_rejected():
    rows = [make_row("a", 1), make_row("b", 2)]
    rows[1]["extra_info"]["index"] = rows[0]["extra_info"]["index"]
    with pytest.raises(ValueError, match="duplicate stable sample ID"):
        select_stratified_rows(rows, total=1)


def test_build_eval_core_writes_dataset_and_auditable_manifest(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rows = [make_row(f"source-{source}", index) for source in range(7) for index in range(3)]
    input_path = tmp_path / "test.parquet"
    output_path = tmp_path / "eval-core-7.jsonl"
    pd.DataFrame(rows).to_parquet(input_path, index=False)

    manifest = build_eval_core(
        input_path,
        output_path,
        total=7,
        seed=42,
        expected_source_count=7,
    )
    written_rows = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    written_manifest = json.loads(
        (tmp_path / "eval-core-7.jsonl.manifest.json").read_text(encoding="utf-8")
    )
    assert len(written_rows) == 7
    assert manifest == written_manifest
    assert manifest["selection_algorithm"] == SELECTION_ALGORITHM
    assert set(manifest["selected_counts"].values()) == {1}
    assert len(manifest["output"]["sha256"]) == 64


def test_builder_detects_source_set_drift(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    input_path = tmp_path / "test.parquet"
    pd.DataFrame([make_row("only-source", index) for index in range(2)]).to_parquet(
        input_path, index=False
    )
    with pytest.raises(ValueError, match="expected 7 data sources"):
        build_eval_core(input_path, tmp_path / "out.jsonl", total=1)
