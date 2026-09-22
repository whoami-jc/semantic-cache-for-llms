"""Compare two fixed cutoffs on existing queries, without aliases or an LLM."""
import json
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from app.cache import SemanticCache
from app.embeddings import RemoteEmbeddings
from app.redis_connection import create_redis_client
from app.support_policy import SupportFAQPolicy
from app.validate_support import cases, summarize


def main():
    source = Path("datasets/store-support.json").read_bytes()
    other = Path("datasets/store-support-evaluation.json").read_bytes()
    dataset = json.loads(source)
    groups = {"development": list(cases(dataset)), "previous_evaluation": json.loads(other)["cases"]}
    client = create_redis_client()
    embeddings = cache = None
    results = {}
    try:
        embeddings = RemoteEmbeddings(os.environ.get("EMBEDDINGS_URL", "http://127.0.0.1:8001"))
        cache = SemanticCache(client, embeddings, namespace="cutoff-check:" + uuid4().hex, policy=SupportFAQPolicy())
        for faq in dataset["faqs"]:
            cache.store(faq["reference"], "store-policy-v1", "placeholder", 0, faq["id"])
        for group, queries in groups.items():
            results[group] = {}
            for cutoff in (0.8, 0.78):
                rows = []
                for query in queries:
                    actual, similarity, elapsed = cache.lookup(query["query"], "store-policy-v1", "placeholder", 0, cutoff)
                    rows.append({**query, "actual": actual, "similarity": similarity, "lookup_ms": elapsed})
                results[group][str(cutoff)] = {"summary": summarize(rows), "cases": rows}
        report = {"development_sha256": sha256(source).hexdigest(), "previous_evaluation_sha256": sha256(other).hexdigest(),
                  "model": embeddings.model_name, "revision": embeddings.revision,
                  "note": "Retrospective comparison on already observed synthetic queries. No aliases, verifier, LLM calls or changes to labels. Not a new held-out evaluation.", "results": results}
    finally:
        if cache is not None:
            cache.clear()
            client.ft(cache.index_name).dropindex(delete_documents=False)
        if embeddings is not None:
            embeddings.close()
        client.close()
    folder = Path("validation/cutoff-comparison")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({g: {t: r["summary"] for t, r in modes.items()} for g, modes in results.items()}, indent=2))
    for group, modes in results.items():
        for before, after in zip(modes["0.8"]["cases"], modes["0.78"]["cases"]):
            if before["actual"] != after["actual"]:
                print(json.dumps({"group": group, "changed": after}))


if __name__ == "__main__":
    main()
