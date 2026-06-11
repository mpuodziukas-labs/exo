"""
Behavioral tests for exo.master.shard_validator.

Focuses on:
- validate() returns ready=True when no shards registered (legacy path)
- validate() returns ready=True when all registered shards are present
- validate() returns ready=False with correct missing list when shards absent
- register_shard() overwrites an existing entry (present → absent)
- mark_all_present() convenience covers every node × shard combination
- get_stats() aggregates totals and present counts per model
- Partial presence: some shards present, some absent
"""
from __future__ import annotations

from exo.master.shard_validator import ShardValidator


class TestShardValidatorValidation:
    def test_no_shards_registered_returns_ready(self) -> None:
        """Legacy path: if no shard info at all, assume ready."""
        sv = ShardValidator()
        result = sv.validate("unknown-model")
        assert result.ready is True
        assert result.total_shards == 0
        assert result.present_shards == 0
        assert result.missing_shards == []

    def test_all_shards_present_returns_ready(self) -> None:
        sv = ShardValidator()
        sv.register_shard("llama3", "node-1", 0, 4, present=True)
        sv.register_shard("llama3", "node-1", 1, 4, present=True)
        sv.register_shard("llama3", "node-1", 2, 4, present=True)
        sv.register_shard("llama3", "node-1", 3, 4, present=True)

        result = sv.validate("llama3")
        assert result.ready is True
        assert len(result.missing_shards) == 0
        assert result.present_shards == 4

    def test_missing_shard_makes_result_not_ready(self) -> None:
        sv = ShardValidator()
        sv.register_shard("llama3", "node-1", 0, 2, present=True)
        sv.register_shard("llama3", "node-1", 1, 2, present=False)  # missing!

        result = sv.validate("llama3")
        assert result.ready is False
        assert ("node-1", 1) in result.missing_shards

    def test_missing_shards_list_completeness(self) -> None:
        """All absent shards from multiple nodes appear in missing_shards."""
        sv = ShardValidator()
        sv.register_shard("model-x", "node-a", 0, 3, present=False)
        sv.register_shard("model-x", "node-a", 1, 3, present=True)
        sv.register_shard("model-x", "node-b", 2, 3, present=False)

        result = sv.validate("model-x")
        missing = set(result.missing_shards)
        assert ("node-a", 0) in missing
        assert ("node-b", 2) in missing
        assert len(result.missing_shards) == 2

    def test_register_shard_overwrites_existing_entry(self) -> None:
        """Re-registering a shard with present=False marks it missing."""
        sv = ShardValidator()
        sv.register_shard("m", "n", 0, 1, present=True)
        sv.register_shard("m", "n", 0, 1, present=False)

        result = sv.validate("m")
        assert result.ready is False

    def test_mark_all_present_marks_every_combination(self) -> None:
        sv = ShardValidator()
        sv.mark_all_present("big-model", node_ids=["n1", "n2"], total_shards=3)

        result = sv.validate("big-model")
        assert result.ready is True
        assert result.present_shards == 6  # 2 nodes × 3 shards

    def test_validate_to_dict_shape(self) -> None:
        sv = ShardValidator()
        sv.register_shard("m", "n", 0, 1, present=True)
        d = sv.validate("m").to_dict()
        assert d["ready"] is True
        assert "missing" in d
        assert "total_shards" in d
        assert "present_shards" in d

    def test_get_stats_aggregates_per_model(self) -> None:
        sv = ShardValidator()
        sv.register_shard("m1", "n", 0, 2, present=True)
        sv.register_shard("m1", "n", 1, 2, present=False)
        sv.register_shard("m2", "n", 0, 1, present=True)

        stats = sv.get_stats()
        by_model = {entry["model_id"]: entry for entry in stats["models"]}
        assert by_model["m1"]["total"] == 2
        assert by_model["m1"]["present"] == 1
        assert by_model["m2"]["total"] == 1
        assert by_model["m2"]["present"] == 1
