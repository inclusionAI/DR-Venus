import json

import pandas as pd
import pytest

from experiments.count_dataset_rows import count_rows


def test_count_rows_supports_parquet_and_nonempty_jsonl(tmp_path):
    parquet = tmp_path / "data.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(parquet, index=False)
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(json.dumps({"x": 1}) + "\n\n" + json.dumps({"x": 2}) + "\n")
    assert count_rows(parquet) == 2
    assert count_rows(jsonl) == 2


def test_count_rows_rejects_unknown_format(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("x")
    with pytest.raises(ValueError, match="parquet or .jsonl"):
        count_rows(path)
