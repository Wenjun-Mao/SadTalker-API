from fastapi.testclient import TestClient

from ingest_nocodb.app import app


client = TestClient(app)


def test_webhook_ok():
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


def test_webhook_empty_rows():
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
