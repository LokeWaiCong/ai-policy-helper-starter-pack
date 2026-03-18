import time, os, hashlib, uuid
from typing import List, Dict, Tuple
import numpy as np
from .settings import settings
from .ingest import chunk_text, doc_hash
from qdrant_client import QdrantClient, models as qm

# ---- Embedders ----
class OpenRouterEmbedder:
    """Uses OpenRouter's text-embedding-3-small API (384-dim)."""
    MODEL = "openai/text-embedding-3-small"

    def __init__(self, api_key: str, dim: int = 384):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(
            model=self.MODEL,
            input=text[:8000],
            dimensions=self.dim,
        )
        v = np.array(resp.data[0].embedding, dtype="float32")
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)


class TFIDFEmbedder:
    """Lightweight corpus-aware TF-IDF embedder — no GPU, no heavy deps."""

    def __init__(self, dim: int = 384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim, stop_words="english", sublinear_tf=True)
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        if not self._fitted:
            self.vectorizer.fit([text])
            self._fitted = True
        v = self.vectorizer.transform([text]).toarray()[0].astype("float32")
        # Pad or truncate to exactly self.dim
        if len(v) < self.dim:
            v = np.pad(v, (0, self.dim - len(v)))
        else:
            v = v[:self.dim]
        norm = np.linalg.norm(v)
        return v / (norm + 1e-9)

# ---- Vector store abstraction ----
class InMemoryStore:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.meta: List[Dict] = []
        self._hashes = set()

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        for v, m in zip(vectors, metadatas):
            h = m.get("hash")
            if h and h in self._hashes:
                continue
            self.vecs.append(v.astype("float32"))
            self.meta.append(m)
            if h:
                self._hashes.add(h)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if not self.vecs:
            return []
        A = np.vstack(self.vecs)  # [N, d]
        q = query.reshape(1, -1)  # [1, d]
        # cosine similarity
        sims = (A @ q.T).ravel() / (np.linalg.norm(A, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9)
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idx]

class QdrantStore:
    def __init__(self, collection: str, dim: int = 384):
        self.client = QdrantClient(url="http://qdrant:6333", timeout=10.0)
        self.collection = collection
        self.dim = dim
        # Retry until qdrant is ready (up to ~30s)
        for attempt in range(15):
            try:
                self._ensure_collection()
                return
            except Exception:
                if attempt < 14:
                    time.sleep(2)
        self._ensure_collection()  # final attempt — raise if still failing

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.collection)
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )

    def upsert(self, vectors: List[np.ndarray], metadatas: List[Dict]):
        points = []
        for i, (v, m) in enumerate(zip(vectors, metadatas)):
            # Qdrant requires unsigned int or UUID — derive a UUID from the SHA-256 hash
            h = m.get("hash") or hashlib.sha256(str(i).encode()).hexdigest()
            point_id = str(uuid.UUID(hex=h[:32]))
            points.append(qm.PointStruct(id=point_id, vector=v.tolist(), payload=m))
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=k,
            with_payload=True
        )
        out = []
        for r in res:
            out.append((float(r.score), dict(r.payload)))
        return out

# ---- LLM provider ----
class StubLLM:
    def generate(self, query: str, contexts: List[Dict]) -> str:
        lines = [f"Answer (stub): Based on the following sources:"]
        for c in contexts:
            sec = c.get("section") or "Section"
            lines.append(f"- {c.get('title')} — {sec}")
        lines.append("Summary:")
        # naive summary of top contexts
        joined = " ".join([c.get("text", "") for c in contexts])
        lines.append(joined[:600] + ("..." if len(joined) > 600 else ""))
        return "\n".join(lines)

class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def generate(self, query: str, contexts: List[Dict]) -> str:
        prompt = f"You are a helpful company policy assistant. Cite sources by title and section when relevant.\nQuestion: {query}\nSources:\n"
        for c in contexts:
            prompt += f"- {c.get('title')} | {c.get('section')}\n{c.get('text')[:600]}\n---\n"
        prompt += "Write a concise, accurate answer grounded in the sources. If unsure, say so."
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )
        return resp.choices[0].message.content

# ---- RAG Orchestrator & Metrics ----
class Metrics:
    def __init__(self):
        self.t_retrieval: List[float] = []
        self.t_generation: List[float] = []
        self.total_queries: int = 0

    def add_retrieval(self, ms: float):
        self.t_retrieval.append(ms)

    def add_generation(self, ms: float):
        self.t_generation.append(ms)
        self.total_queries += 1

    def summary(self) -> Dict:
        avg_r = sum(self.t_retrieval) / len(self.t_retrieval) if self.t_retrieval else 0.0
        avg_g = sum(self.t_generation) / len(self.t_generation) if self.t_generation else 0.0
        return {
            "total_queries": self.total_queries,
            "avg_retrieval_latency_ms": round(avg_r, 2),
            "avg_generation_latency_ms": round(avg_g, 2),
        }

class RAGEngine:
    def __init__(self):
        # Always use TF-IDF for embeddings (fully offline, no API key needed)
        self.embedder = TFIDFEmbedder(dim=384)
        self.embedding_name = "tfidf-384"

        # Vector store selection
        if settings.vector_store == "qdrant":
            try:
                self.store = QdrantStore(collection=settings.collection_name, dim=384)
            except Exception:
                self.store = InMemoryStore(dim=384)
        else:
            self.store = InMemoryStore(dim=384)

        # LLM selection
        if settings.llm_provider == "openrouter" and settings.openrouter_api_key:
            try:
                self.llm = OpenRouterLLM(
                    api_key=settings.openrouter_api_key,
                    model=settings.llm_model,
                )
                self.llm_name = f"openrouter:{settings.llm_model}"
            except Exception:
                self.llm = StubLLM()
                self.llm_name = "stub"
        else:
            self.llm = StubLLM()
            self.llm_name = "stub"

        self.metrics = Metrics()
        self._doc_titles = set()
        self._chunk_count = 0

    def ingest_chunks(self, chunks: List[Dict]) -> Tuple[int, int]:
        vectors = []
        metas = []
        doc_titles_before = set(self._doc_titles)

        # Fit TF-IDF on the full corpus before embedding
        if isinstance(self.embedder, TFIDFEmbedder):
            self.embedder.fit([ch["text"] for ch in chunks])

        for ch in chunks:
            text = ch["text"]
            h = doc_hash(text)
            meta = {
                "id": h,
                "hash": h,
                "title": ch["title"],
                "section": ch.get("section"),
                "text": text,
            }
            v = self.embedder.embed(text)
            vectors.append(v)
            metas.append(meta)
            self._doc_titles.add(ch["title"])
            self._chunk_count += 1

        self.store.upsert(vectors, metas)
        return (len(self._doc_titles) - len(doc_titles_before), len(metas))

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        t0 = time.time()
        qv = self.embedder.embed(query)
        results = self.store.search(qv, k=k)
        self.metrics.add_retrieval((time.time()-t0)*1000.0)
        return [meta for score, meta in results]

    def generate(self, query: str, contexts: List[Dict]) -> str:
        t0 = time.time()
        answer = self.llm.generate(query, contexts)
        self.metrics.add_generation((time.time()-t0)*1000.0)
        return answer

    def stats(self) -> Dict:
        m = self.metrics.summary()
        return {
            "total_docs": len(self._doc_titles),
            "total_chunks": self._chunk_count,
            "embedding_model": self.embedding_name,
            "llm_model": self.llm_name,
            **m,
        }

# ---- Helpers ----
def build_chunks_from_docs(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            out.append({"title": d["title"], "section": d["section"], "text": ch})
    return out
