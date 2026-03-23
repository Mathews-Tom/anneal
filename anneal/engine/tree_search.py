"""UCB tree search over artifact space for backtracking and branch exploration."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from anneal.engine.types import Direction, EvalResult, ExperimentRecord

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the search tree. Maps 1:1 to a git commit."""

    sha: str
    score: float
    visit_count: int = 0
    children: list[TreeNode] = field(default_factory=list)
    parent_sha: str | None = None
    pruned: bool = False


class UCBTreeSearch:
    """Tree-structured search using Upper Confidence Bound selection.

    Implements SearchStrategy protocol for integration with StrategySelector.
    Uses UCB1 to balance exploitation (high-scoring nodes) with exploration
    (under-visited nodes). Supports pruning, persistence, and bootstrap
    from experiment history.
    """

    def __init__(
        self,
        exploration_constant: float = 1.414,
        max_depth: int = 50,
        prune_threshold: int = 5,
    ) -> None:
        self._c = exploration_constant
        self._max_depth = max_depth
        self._prune_threshold = prune_threshold
        self._root: TreeNode | None = None
        self._nodes: dict[str, TreeNode] = {}
        self._total_visits: int = 0

    @property
    def root(self) -> TreeNode | None:
        return self._root

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    def _ucb1(self, node: TreeNode) -> float:
        """Compute UCB1 value for a node."""
        if node.visit_count == 0:
            return float("inf")
        if self._total_visits == 0:
            return node.score
        exploitation = node.score
        exploration = self._c * math.sqrt(math.log(self._total_visits) / node.visit_count)
        return exploitation + exploration

    def _add_node(self, sha: str, score: float, parent_sha: str | None = None) -> TreeNode:
        """Add a node to the tree. If it exists, update score."""
        if sha in self._nodes:
            node = self._nodes[sha]
            node.score = max(node.score, score)  # Keep best score seen
            return node
        node = TreeNode(sha=sha, score=score, parent_sha=parent_sha)
        self._nodes[sha] = node
        if parent_sha and parent_sha in self._nodes:
            parent = self._nodes[parent_sha]
            if node not in parent.children:
                parent.children.append(node)
        if self._root is None:
            self._root = node
        return node

    def select_parent(self) -> str:
        """Select the node to mutate next via UCB1.

        Returns the git SHA of the selected parent.
        Considers only non-pruned leaf and internal nodes.
        """
        if not self._nodes:
            raise ValueError("Tree is empty — cannot select parent")

        candidates = [
            node for node in self._nodes.values()
            if not node.pruned and self._depth(node) < self._max_depth
        ]
        if not candidates:
            # All nodes pruned — fall back to root
            if self._root is not None:
                return self._root.sha
            raise ValueError("Tree is empty — cannot select parent")

        best = max(candidates, key=self._ucb1)
        return best.sha

    def record_outcome(
        self,
        parent_sha: str,
        child_sha: str,
        score: float,
        kept: bool,
    ) -> None:
        """Update the tree with a new experiment outcome."""
        # Ensure parent exists
        if parent_sha not in self._nodes:
            self._add_node(parent_sha, score=0.0)

        # Add child
        child = self._add_node(child_sha, score, parent_sha=parent_sha)

        # Update visit counts
        child.visit_count += 1
        self._total_visits += 1

        # Propagate visit to ancestors
        current_sha = parent_sha
        while current_sha is not None and current_sha in self._nodes:
            self._nodes[current_sha].visit_count += 1
            current_sha = self._nodes[current_sha].parent_sha

        # Auto-prune check on parent subtree
        if not kept:
            self._check_prune(parent_sha)

    def _check_prune(self, sha: str) -> None:
        """Prune a node if it has too many consecutive non-improvements."""
        if sha not in self._nodes:
            return
        node = self._nodes[sha]
        non_improving = sum(
            1 for child in node.children
            if child.score <= node.score
        )
        if non_improving >= self._prune_threshold and len(node.children) >= self._prune_threshold:
            self.prune_subtree(sha)

    def prune_subtree(self, sha: str) -> None:
        """Mark a subtree as pruned. Pruned nodes are never selected."""
        if sha not in self._nodes:
            return
        node = self._nodes[sha]
        # Don't prune root
        if node is self._root:
            return
        node.pruned = True
        for child in node.children:
            self.prune_subtree(child.sha)

    def should_keep(
        self,
        challenger_result: EvalResult,
        baseline_score: float,
        baseline_raw_scores: list[float] | None,
        direction: Direction,
        min_improvement_threshold: float = 0.0,
        confidence: float = 0.95,
    ) -> bool:
        """SearchStrategy protocol — accept if improving in the given direction."""
        if direction is Direction.HIGHER_IS_BETTER:
            return challenger_result.score > baseline_score + min_improvement_threshold
        return challenger_result.score < baseline_score - min_improvement_threshold

    def bootstrap_from_history(self, records: list[ExperimentRecord]) -> None:
        """Reconstruct tree from existing experiment records.

        Uses pre_experiment_sha -> git_sha edges to rebuild the DAG.
        """
        for record in records:
            parent_sha = record.pre_experiment_sha
            child_sha = record.git_sha

            if parent_sha not in self._nodes:
                self._add_node(parent_sha, score=record.baseline_score)

            self._add_node(child_sha, score=record.score, parent_sha=parent_sha)
            self._nodes[child_sha].visit_count += 1
            self._total_visits += 1

    def _depth(self, node: TreeNode) -> int:
        """Compute depth of a node from root."""
        depth = 0
        current = node
        while current.parent_sha is not None and current.parent_sha in self._nodes:
            depth += 1
            current = self._nodes[current.parent_sha]
            if depth > self._max_depth:
                break
        return depth

    def get_tree_info(self) -> dict[str, int]:
        """Return tree statistics for context injection."""
        if not self._nodes:
            return {"nodes": 0, "depth": 0, "pruned": 0}
        max_depth = max(self._depth(n) for n in self._nodes.values())
        pruned_count = sum(1 for n in self._nodes.values() if n.pruned)
        return {
            "nodes": len(self._nodes),
            "depth": max_depth,
            "pruned": pruned_count,
            "total_visits": self._total_visits,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: Path) -> None:
        """Serialize tree state to JSON for crash recovery."""
        data = {
            "exploration_constant": self._c,
            "max_depth": self._max_depth,
            "prune_threshold": self._prune_threshold,
            "total_visits": self._total_visits,
            "nodes": {
                sha: {
                    "sha": node.sha,
                    "score": node.score,
                    "visit_count": node.visit_count,
                    "parent_sha": node.parent_sha,
                    "pruned": node.pruned,
                    "children": [c.sha for c in node.children],
                }
                for sha, node in self._nodes.items()
            },
            "root_sha": self._root.sha if self._root else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> UCBTreeSearch:
        """Restore tree from persisted state."""
        data = json.loads(path.read_text())
        tree = cls(
            exploration_constant=data.get("exploration_constant", 1.414),
            max_depth=data.get("max_depth", 50),
            prune_threshold=data.get("prune_threshold", 5),
        )
        tree._total_visits = data.get("total_visits", 0)

        # First pass: create all nodes
        for sha, node_data in data.get("nodes", {}).items():
            node = TreeNode(
                sha=node_data["sha"],
                score=node_data["score"],
                visit_count=node_data.get("visit_count", 0),
                parent_sha=node_data.get("parent_sha"),
                pruned=node_data.get("pruned", False),
            )
            tree._nodes[sha] = node

        # Second pass: link children
        for sha, node_data in data.get("nodes", {}).items():
            node = tree._nodes[sha]
            for child_sha in node_data.get("children", []):
                if child_sha in tree._nodes:
                    node.children.append(tree._nodes[child_sha])

        # Set root
        root_sha = data.get("root_sha")
        if root_sha and root_sha in tree._nodes:
            tree._root = tree._nodes[root_sha]

        return tree
