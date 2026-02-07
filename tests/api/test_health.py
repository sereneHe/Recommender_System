import pytest

pytestmark = pytest.mark.asyncio


async def test_health(api_client):
    r = await api_client.get("/health")
    assert r.status_code == 200 # 200 = OK!
    assert r.json() == {"status": "healthy"}