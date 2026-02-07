from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


def to_csv(rows: list[dict]) -> bytes:
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for row in rows:
        lines.append(",".join(str(row[c]) for c in cols))
    return ("\n".join(lines) + "\n").encode("utf-8")

# Test cases for the /evaluate-csv endpoint
async def test_evaluate_csv_no_labels(api_client):
    csv_bytes = to_csv(
        [
            {"f1": 1, "f2": 2, "f3": 3},
            {"f1": -1, "f2": -2, "f3": -3},
        ]
    )
    files = {"file": ("data.csv", csv_bytes, "text/csv")}
    r = await api_client.post("/evaluate-csv", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["has_labels"] is False
    assert body["n_samples"] == 2
    assert body["n_features"] == 3


async def test_missing_columns(api_client):
    csv_bytes = to_csv([{"f1": 1, "f2": 2}])  # missing f3
    files = {"file": ("data.csv", csv_bytes, "text/csv")}
    r = await api_client.post("/evaluate-csv", files=files)
    assert r.status_code == 400
    assert "Missing columns" in r.json()["detail"]