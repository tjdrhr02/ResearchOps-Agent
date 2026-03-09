from fastapi.testclient import TestClient

from src.main import app


def test_run_research_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/research/run",
        json={"user_query": "RAG evaluation in production", "max_sources": 4},
    )
    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert body["status"] == "completed"
    assert body["brief"]["executive_summary"]


def test_get_research_job_and_sources() -> None:
    client = TestClient(app)
    run_response = client.post(
        "/research/run",
        json={"user_query": "Agentic workflow design", "max_sources": 3},
    )
    job_id = run_response.json()["job_id"]

    job_response = client.get(f"/research/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json()["job"]["job_id"] == job_id

    sources_response = client.get(f"/research/{job_id}/sources")
    assert sources_response.status_code == 200
    assert sources_response.json()["job_id"] == job_id
    assert isinstance(sources_response.json()["sources"], list)


def test_notes_save_and_search() -> None:
    client = TestClient(app)
    save_response = client.post("/notes/save", json={"note": "RAG latency tuning with cache"})
    assert save_response.status_code == 200
    assert save_response.json()["note_id"]

    search_response = client.get("/notes/search", params={"keyword": "latency"})
    assert search_response.status_code == 200
    assert any("latency" in item.lower() for item in search_response.json()["items"])


def test_metrics_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
