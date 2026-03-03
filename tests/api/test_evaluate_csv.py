from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from hei_project import api


def test_recommend_returns_stubbed_result(api_app):
    body = asyncio.run(api.recommend_json({}))
    assert body["targets"] == ["GLU (mg/dL)"]
    assert body["results"]["GLU (mg/dL)"]["selected_features"] == ["food_a", "bio_a"]
    assert body["results"]["GLU (mg/dL)"]["test_errors"] == [0.3, 0.4]


def test_recommend_rejects_empty_targets(api_app):
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(api.recommend_json({"targets": []}))

    assert exc_info.value.status_code == 400
    assert "non-empty list" in exc_info.value.detail
