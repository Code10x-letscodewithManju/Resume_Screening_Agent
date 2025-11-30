from typing import List
import numpy as np
from langchain_openai import OpenAIEmbeddings

from .config import OPENAI_EMBED_MODEL


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype="float32")
    emb_model = get_embedding_model()
    vectors = emb_model.embed_documents(texts)
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    arr = arr / norms
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b.T))
