"""Evaluate a warm store FAQ cache using real Redis and Docker models, no paid LLM."""
import json
import os
from hashlib import sha256
from pathlib import Path
from statistics import median
from uuid import uuid4

from app.cache import SemanticCache
from app.embeddings import RemoteEmbeddings
from app.redis_connection import create_redis_client
from app.support_policy import SupportFAQPolicy, RETRIEVAL_THRESHOLD


def cases(dataset):
    for faq in dataset["faqs"]:
        yield {"query": faq["reference"], "type": "exact", "expected": faq["id"]}
        for query in faq["paraphrases"]:
            yield {"query": query, "type": "paraphrase", "expected": faq["id"]}
        for query in faq["negatives"]:
            yield {"query": query, "type": "hard_negative", "expected": None}
    for query in dataset["bypass"]:
        yield {"query": query, "type": "bypass", "expected": None}


def summarize(rows):
    correct = sum(r["actual"] == r["expected"] and r["actual"] is not None for r in rows)
    wrong = sum(r["actual"] is not None and r["actual"] != r["expected"] for r in rows)
    positives = sum(r["expected"] is not None for r in rows)
    return {
        "cases": len(rows), "correct_reuse": correct, "wrong_reuse": wrong,
        "positive_cases": positives, "missed_positives": positives - correct,
        "precision": correct / (correct + wrong) if correct + wrong else None,
        "decision_accuracy": sum(r["actual"] == r["expected"] for r in rows) / len(rows),
        "paraphrase_hits": sum(r["type"] == "paraphrase" and r["actual"] == r["expected"] for r in rows),
        "paraphrases": sum(r["type"] == "paraphrase" for r in rows),
        "bypass_misses": sum(r["type"] == "bypass" and r["actual"] is None for r in rows),
        "median_lookup_ms": median(r["lookup_ms"] for r in rows),
    }


def main():
    data = Path("datasets/store-support.json").read_bytes()
    dataset = json.loads(data)
    url = os.environ.get("EMBEDDINGS_URL", "http://127.0.0.1:8001")
    client = create_redis_client()
    embeddings = None
    caches = []
    output = {}
    try:
        embeddings = RemoteEmbeddings(url)
        for mode, threshold, gate, policy in [
            ("baseline", 0.9, None, None),
            ("store_support", RETRIEVAL_THRESHOLD, None, SupportFAQPolicy()),
        ]:
            cache = SemanticCache(client, embeddings, threshold=threshold,
                                  policy=policy, namespace="support-validation:" + uuid4().hex)
            caches.append(cache)
            for faq in dataset["faqs"]:
                cache.store(faq["reference"], "store-policy-v1", "placeholder", 0, faq["id"])
            rows = []
            for case in cases(dataset):
                actual, similarity, elapsed = cache.lookup(case["query"], "store-policy-v1", "placeholder", 0)
                rows.append({**case, "actual": actual, "similarity": similarity, "lookup_ms": elapsed})
            output[mode] = {"summary": summarize(rows), "cases": rows}
        report = {
            "dataset_sha256": sha256(data).hexdigest(), "embedding_model": embeddings.model_name,
            "embedding_revision": embeddings.revision, "verifier_model": None,
            "retrieval_threshold": RETRIEVAL_THRESHOLD,
            "policy_revision": SupportFAQPolicy.revision, "results": output,
        }
    finally:
        for cache in caches:
            cache.clear()
            client.ft(cache.index_name).dropindex(delete_documents=False)
        if embeddings is not None:
            embeddings.close()
        client.close()
    folder = Path("validation/store-support")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({mode: result["summary"] for mode, result in output.items()}, indent=2))


if __name__ == "__main__":
    main()
