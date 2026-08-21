import json
from pathlib import Path

import pandas as pd

from drvenus_repro.manifests import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_committed_eval_core_matches_source_and_manifest():
    cohort = ROOT / "evaluation/data/eval-core-100.parquet"
    manifest = load_manifest(cohort.with_name(cohort.name + ".manifest.json"))
    source = ROOT / "RL/data/test.parquet"
    frame = pd.read_parquet(cohort)

    assert manifest["input"]["sha256"] == sha256_file(source)
    assert manifest["output"]["sha256"] == sha256_file(cohort)
    assert len(frame) == manifest["num_samples"] == 100
    assert frame["data_source"].value_counts().sort_index().to_dict() == manifest["selected_counts"]
    ids = [extra["eval_sample_id"] for extra in frame["extra_info"]]
    assert ids == manifest["sample_ids"]
    assert len(set(ids)) == 100


def test_committed_pass4_subset_matches_frozen_eval_core():
    cohort = ROOT / "evaluation/data/pass4-20.parquet"
    manifest_path = cohort.with_name(cohort.name + ".manifest.json")
    manifest = load_manifest(manifest_path)
    parent = ROOT / "evaluation/data/eval-core-100.parquet"
    parent_manifest = parent.with_name(parent.name + ".manifest.json")
    frame = pd.read_parquet(cohort)

    assert manifest["parent"]["cohort_sha256"] == sha256_file(parent)
    assert manifest["parent"]["manifest_sha256"] == sha256_file(parent_manifest)
    assert manifest["output"]["sha256"] == sha256_file(cohort)
    assert len(frame) == manifest["num_samples"] == 20
    assert frame["data_source"].value_counts().sort_index().to_dict() == manifest["selected_counts"]
    ids = [extra["eval_sample_id"] for extra in frame["extra_info"]]
    assert ids == manifest["sample_ids"]
    assert len(set(ids)) == 20


def test_committed_rl_subset_matches_source_and_manifest():
    cohort = ROOT / "experiments/data/rl-train-96.parquet"
    manifest = load_manifest(cohort.with_name(cohort.name + ".manifest.json"))
    source = ROOT / "RL/data/train.parquet"
    frame = pd.read_parquet(cohort)

    assert manifest["input"]["sha256"] == sha256_file(source)
    assert manifest["output"]["sha256"] == sha256_file(cohort)
    assert len(frame) == manifest["num_samples"] == 96
    assert frame["data_source"].value_counts().sort_index().to_dict() == manifest["selected_counts"]
    ids = [extra["rl_question_id"] for extra in frame["extra_info"]]
    assert ids == manifest["question_ids"]
    assert len(set(ids)) == 96
    assert manifest["duplicate_rows_removed"] == 3


def test_committed_rl_validation_excludes_training_questions():
    cohort = ROOT / "experiments/data/rl-val-32.parquet"
    manifest = load_manifest(cohort.with_name(cohort.name + ".manifest.json"))
    train = ROOT / "experiments/data/rl-train-96.parquet"
    frame = pd.read_parquet(cohort)

    assert manifest["excluded_train"]["sha256"] == sha256_file(train)
    assert manifest["output"]["sha256"] == sha256_file(cohort)
    assert len(frame) == manifest["num_samples"] == 32
    assert frame["data_source"].value_counts().sort_index().to_dict() == manifest["selected_counts"]
    assert all(extra["rl_subset"] == "rl-val-32" for extra in frame["extra_info"])
