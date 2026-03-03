from hei_project import api


def test_health_returns_current_status_payload(api_app):
    body = api.health_check()
    assert body["status"] == "healthy"
    assert body["loaded"] is True
    assert "assets" in body
