def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_shape(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    for key in ("total_docs", "total_chunks", "total_queries",
                "avg_retrieval_latency_ms", "avg_generation_latency_ms",
                "embedding_model", "llm_model"):
        assert key in data, f"missing key: {key}"


def test_ingest(client):
    r = client.post("/api/ingest")
    assert r.status_code == 200
    data = r.json()
    assert data["indexed_chunks"] > 0
    assert data["indexed_docs"] > 0


def test_ask_returns_citations(client):
    # Ensure docs are ingested first
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query": "What is the refund window for small appliances?"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["answer"], str) and len(data["answer"]) > 0
    assert len(data["citations"]) > 0
    assert len(data["chunks"]) > 0
    # Every citation must have a title
    for c in data["citations"]:
        assert c["title"]


def test_ask_returns_chunk_text(client):
    client.post("/api/ingest")
    r = client.post("/api/ask", json={"query": "shipping SLA to East Malaysia"})
    assert r.status_code == 200
    data = r.json()
    # Chunks must contain non-empty text
    for ch in data["chunks"]:
        assert ch["text"].strip()


def test_metrics_queries_increment(client):
    r0 = client.get("/api/metrics").json()
    before = r0["total_queries"]
    client.post("/api/ask", json={"query": "warranty coverage"})
    r1 = client.get("/api/metrics").json()
    assert r1["total_queries"] == before + 1
