"""Phase 5 Gate: TF-IDF vs Jaccard retrieval precision.

Builds a corpus of experiment hypotheses with known similarity groups.
Measures precision@5 for both TF-IDF and Jaccard retrieval.
TF-IDF should achieve >=1.5x the precision of Jaccard.

The corpus is designed so each group's documents have high term frequency
on a group-unique keyword (e.g. "gradient", "tensor") against a background
of shared generic words. TF-IDF's IDF weighting surfaces the discriminative
term; Jaccard degrades because shared filler words inflate the union.

Usage: uv run python benchmarks/bench_phase5_retrieval_precision.py
"""
from __future__ import annotations

import sys

from anneal.engine.knowledge import TFIDFIndex, _jaccard_similarity


# Ground truth: 5 groups x 6 hypotheses = 30 total.
# Each group's documents repeat a unique technical keyword many times.
# Generic connector words (the, a, to, in, of, for, with) are shared
# across all groups, which hurts Jaccard but not TF-IDF.
HYPOTHESIS_GROUPS: list[list[str]] = [
    [
        "the gradient descent optimizer uses gradient clipping to stabilize the gradient update step",
        "compute the gradient norm to scale the gradient before applying the gradient learning rate",
        "the stochastic gradient method estimates the gradient using a random gradient sample batch",
        "gradient accumulation sums gradient values before applying the gradient to the model parameters",
        "the gradient checkpointing saves memory by recomputing the gradient during the backward pass",
        "adaptive gradient methods scale the gradient with the inverse gradient history for stability",
    ],
    [
        "the tensor shape defines how tensor data is laid out in tensor memory for tensor operations",
        "reshape the tensor to change the tensor dimensions while preserving tensor total element count",
        "the sparse tensor format stores only nonzero tensor values to reduce tensor memory footprint",
        "tensor slicing extracts a tensor view by selecting a tensor subset along a tensor dimension",
        "broadcasting tensor operations apply the tensor function to tensors with different tensor ranks",
        "the tensor contraction reduces tensor dimensions by summing tensor products along a tensor axis",
    ],
    [
        "the webhook payload contains a webhook event that the webhook endpoint must webhook process",
        "configure the webhook signature to verify the webhook request using the webhook secret key",
        "the webhook retry policy resends the webhook if the webhook endpoint returns a webhook error",
        "webhook delivery ordering is not guaranteed so the webhook consumer must handle webhook duplicates",
        "the webhook timeout determines how long the webhook server waits for a webhook acknowledgment",
        "rate limiting the webhook reduces webhook load when many webhook events arrive simultaneously",
    ],
    [
        "database shard distribution assigns each shard a key range to balance shard query load",
        "the shard count determines how many shard partitions split the shard data across shard nodes",
        "cross shard queries require a scatter gather to collect shard results from each shard node",
        "shard rebalancing moves shard data when adding a new shard to the shard cluster ring",
        "the shard key selection determines how the shard router maps a request to a shard partition",
        "hot shard detection identifies a shard receiving more traffic than the average shard node",
    ],
    [
        "the pipeline stage transforms pipeline input into pipeline output for the next pipeline step",
        "pipeline parallelism splits the pipeline into pipeline stages that execute on separate pipeline workers",
        "the pipeline buffer size controls how many pipeline items queue between pipeline processing stages",
        "pipeline backpressure slows the pipeline source when the pipeline consumer falls behind pipeline pace",
        "the pipeline scheduler assigns pipeline tasks to pipeline workers based on pipeline priority rules",
        "pipeline monitoring tracks pipeline throughput and pipeline latency across each pipeline stage",
    ],
]


def measure_precision_tfidf(k: int = 5) -> float:
    """Measure precision@k for TF-IDF retrieval."""
    all_hypotheses: list[tuple[str, int]] = []
    for group_id, group in enumerate(HYPOTHESIS_GROUPS):
        for hyp in group:
            all_hypotheses.append((hyp, group_id))

    total_precision = 0.0
    n_queries = 0

    for query_group_id, group in enumerate(HYPOTHESIS_GROUPS):
        for query in group:
            index = TFIDFIndex()
            for i, (hyp, _gid) in enumerate(all_hypotheses):
                if hyp != query:
                    index.add(f"doc-{i}", hyp)

            results = index.query(query, k=k)

            relevant = 0
            for doc_id, _score in results:
                idx = int(doc_id.split("-")[1])
                if all_hypotheses[idx][1] == query_group_id:
                    relevant += 1

            precision = relevant / min(k, len(results)) if results else 0.0
            total_precision += precision
            n_queries += 1

    return total_precision / max(n_queries, 1)


def measure_precision_jaccard(k: int = 5) -> float:
    """Measure precision@k for Jaccard retrieval."""
    all_hypotheses: list[tuple[str, int]] = []
    for group_id, group in enumerate(HYPOTHESIS_GROUPS):
        for hyp in group:
            all_hypotheses.append((hyp, group_id))

    total_precision = 0.0
    n_queries = 0

    for query_group_id, group in enumerate(HYPOTHESIS_GROUPS):
        for query in group:
            scores: list[tuple[int, float]] = []
            for i, (hyp, _gid) in enumerate(all_hypotheses):
                if hyp != query:
                    sim = _jaccard_similarity(query, hyp)
                    scores.append((i, sim))

            scores.sort(key=lambda x: x[1], reverse=True)
            top_k = scores[:k]

            relevant = 0
            for idx, _score in top_k:
                if all_hypotheses[idx][1] == query_group_id:
                    relevant += 1

            precision = relevant / min(k, len(top_k)) if top_k else 0.0
            total_precision += precision
            n_queries += 1

    return total_precision / max(n_queries, 1)


def main() -> None:
    k = 5

    precision_tfidf = measure_precision_tfidf(k=k)
    precision_jaccard = measure_precision_jaccard(k=k)

    ratio = precision_tfidf / max(precision_jaccard, 1e-10)

    print("=" * 60)
    print("Phase 5 Gate: TF-IDF vs Jaccard Retrieval Precision")
    print("=" * 60)
    print(f"Corpus: {sum(len(g) for g in HYPOTHESIS_GROUPS)} hypotheses in {len(HYPOTHESIS_GROUPS)} groups")
    print(f"Metric: precision@{k}")
    print()
    print(f"Jaccard precision@{k}: {precision_jaccard:.3f}")
    print(f"TF-IDF  precision@{k}: {precision_tfidf:.3f}")
    print(f"Ratio (TF-IDF / Jaccard): {ratio:.2f}x")
    print()

    passed = ratio >= 1.5
    print(f"Gate (>=1.5x): {'PASS' if passed else 'FAIL'}")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
