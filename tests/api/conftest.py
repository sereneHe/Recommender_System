from __future__ import annotations

import numpy as np
import pytest
import torch
import httpx

from mlo_group_project import api


class DummyScaler:
    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class DummyLabelEncoder:
    def transform(self, y):
        mapping = {"B": 0, "M": 1}
        return np.array([mapping[v] for v in y], dtype=np.int64)


class DummyGuard:
    def validate(self, x: torch.Tensor) -> bool:
        return True


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


@pytest.fixture
def api_app(monkeypatch):
    monkeypatch.setattr(api, "DataGuard", lambda: DummyGuard())

    api.app.state.feature_columns = ["f1", "f2", "f3"]
    api.app.state.scaler = DummyScaler()
    api.app.state.label_encoder = DummyLabelEncoder()
    api.app.state.model = DummyModel()

    return api.app


@pytest.fixture
async def api_client(api_app):
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        yield client