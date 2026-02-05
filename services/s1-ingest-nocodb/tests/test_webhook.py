from fastapi.testclient import TestClient

from ingest_nocodb.app import app
from ingest_nocodb import messaging


client = TestClient(app)


def _noop_enqueue(*_args, **_kwargs):
    return None


def test_webhook_ok(monkeypatch):
    monkeypatch.setenv("S1_RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv("S1_NOCODB_API_KEY", "test")
    monkeypatch.setenv("S1_NOCODB_BASE_URL", "https://example.com")
    monkeypatch.setattr("ingest_nocodb.app.enqueue_downstream", _noop_enqueue)
    payload = {
        "type": "records.after.trigger",
        "id": "074d261a-66c4-4c05-99af-3ffb6b964f7a",
        "version": "v3",
        "data": {
            "table_id": "murmwwnt5ukvp7i",
            "table_name": "创意素材",
            "rows": [
                {
                    "Id": 1,
                    "CreatedAt": "2026-01-27T03:21:31.929Z",
                    "UpdatedAt": "2026-01-27T03:21:31.929Z",
                    "Title": "Sample Text",
                    "url": "Sample Text",
                    "content": "Sample Text",
                    "originaltext": "Sample Text",
                    "image1": "Sample Text",
                    "image2": "Sample Text",
                    "image3": "Sample Text",
                    "image4": "Sample Text",
                    "image5": "Sample Text",
                    "fengmianimage": "Sample Text",
                    "chengpinurl": "Sample Text",
                }
            ],
        },
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 200
    assert response.json() == {"ok": True, "received_rows": 1}


def test_webhook_empty_rows(monkeypatch):
    monkeypatch.setenv("S1_RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv("S1_NOCODB_API_KEY", "test")
    monkeypatch.setenv("S1_NOCODB_BASE_URL", "https://example.com")
    monkeypatch.setattr("ingest_nocodb.app.enqueue_downstream", _noop_enqueue)
    payload = {
        "type": "records.after.trigger",
        "id": "074d261a-66c4-4c05-99af-3ffb6b964f7a",
        "version": "v3",
        "data": {
            "table_id": "murmwwnt5ukvp7i",
            "table_name": "创意素材",
            "rows": [],
        },
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "No rows provided in payload."


def test_webhook_missing_fields(monkeypatch):
    monkeypatch.setenv("S1_RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv("S1_NOCODB_API_KEY", "test")
    monkeypatch.setenv("S1_NOCODB_BASE_URL", "https://example.com")
    monkeypatch.setattr("ingest_nocodb.app.enqueue_downstream", _noop_enqueue)

    payload = {
        "type": "records.after.trigger",
        "id": "074d261a-66c4-4c05-99af-3ffb6b964f7a",
        "version": "v3",
        "data": {
            "table_id": "murmwwnt5ukvp7i",
            "table_name": "创意素材",
            "rows": [
                {
                    "Id": 1,
                    "CreatedAt": "2026-01-27T03:21:31.929Z",
                    "UpdatedAt": "2026-01-27T03:21:31.929Z",
                    "Title": "Sample Text",
                    "url": "Sample Text",
                    "content": "Sample Text",
                    "originaltext": "",
                }
            ],
        },
    }

    response = client.post("/webhook", json=payload)
    assert response.status_code == 400
    assert "missing mandatory fields: originaltext" in response.json()["detail"]
