import json

from drvenus_repro.manifests import load_experiment_manifest, sha256_file, write_predictions
from experiments.create_manifest import build_manifest


def test_manifest_is_built_from_real_hashed_artifacts(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {"sample_id": "one", "data_source": "nq", "references": ["a"]}
        )
        + "\n",
        encoding="utf-8",
    )
    tool_cache = tmp_path / "tool-cache.jsonl"
    tool_cache.write_text('{}\n', encoding="utf-8")
    judge_cache = tmp_path / "judge-cache.jsonl"
    judge_cache.write_text(
        json.dumps(
            {
                "cache_key": "judge-one",
                "model": "judge-model",
                "provider": "openai-compatible",
                "prompt_version": "answer-correctness-v1",
                "response_sha256": "d" * 64,
                "score": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    write_predictions(
        predictions,
        [
            {
                "schema_version": "1.0",
                "sample_id": "one",
                "data_source": "nq",
                "model": "model",
                "seed": 42,
                "sample_index": 0,
                "answer": "a",
                "references": ["a"],
                "score": 1.0,
                "completed": True,
                "format_valid": True,
                "turns": 1,
                "tool_calls": 0,
                "browse_calls": 0,
                "timeout_count": 0,
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_seconds": 1.0,
                "cost_cny": 0.0,
                "max_turns": 12,
                "context_length": 32768,
                "trajectory_path": None,
                "metadata": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "tool_timeout_seconds": 150.0,
                    "max_tokens_per_turn": 1024,
                    "sampling_seed_strategy": "base+sample_index*1000003+llm_call_index",
                    "sampling_seed": 42,
                    "llm_call_count": 1,
                    "llm_sampling_seeds": [42],
                    "tool_cache_mode": "replay",
                    "tool_cache_sha256": sha256_file(tool_cache),
                    "checkpoint_revision": "b" * 40,
                    "checkpoint_sha256": None,
                    "model_revision": "b" * 40,
                    "run_fingerprint": "f" * 64,
                    "judge": {
                        "cache_key": "judge-one",
                        "model": "judge-model",
                        "provider": "openai-compatible",
                        "prompt_version": "answer-correctness-v1",
                        "response_sha256": "d" * 64,
                    },
                },
            }
        ],
    )
    output = tmp_path / "run/experiment_manifest.yaml"

    manifest = build_manifest(
        output_path=output,
        experiment_id="test",
        checkpoint="checkpoint",
        model="model",
        evaluation_dataset=dataset,
        predictions=predictions,
        seeds=[42],
        gpu_type="A100-80G",
        gpu_count=2,
        gpu_hours=0.1,
        api_cost_cny=0.0,
        max_turns=12,
        context_length=32768,
        tool_cache_mode="replay",
        tool_cache=tool_cache,
        judge_cache=judge_cache,
        judge_model="judge-model",
        upstream_commit="a" * 40,
        checkpoint_revision="b" * 40,
        allow_dirty=True,
    )

    loaded = load_experiment_manifest(output)
    assert loaded == manifest
    assert loaded.evaluation_dataset["path"].startswith("../")
    assert loaded.protocol["official_comparable"] is False
    assert loaded.config["checkpoint_revision"] == "b" * 40
    assert loaded.config["repository"]["upstream_base_commit"] == "a" * 40


def test_replay_manifest_requires_cache_artifact(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"sample_id": "one"}) + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("{}\n", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="tool_cache"):
        build_manifest(
            output_path=tmp_path / "manifest.yaml",
            experiment_id="test",
            checkpoint="checkpoint",
            checkpoint_revision="revision",
            model="model",
            evaluation_dataset=dataset,
            predictions=predictions,
            seeds=[42],
            gpu_type="A100-80G",
            gpu_count=2,
            gpu_hours=0.1,
            api_cost_cny=0.0,
            max_turns=12,
            context_length=32768,
            tool_cache_mode="replay",
            upstream_commit="a" * 40,
            allow_dirty=True,
        )
