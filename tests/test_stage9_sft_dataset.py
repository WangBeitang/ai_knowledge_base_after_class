import json

import pytest

from evaluation.stage9.model_planner.prompt_builder import build_planner_prompt
from evaluation.stage9.model_planner.sft_dataset import load_sft_train_examples


def test_stage9_sft_dataset_loads_existing_training_seed():
    examples, stats = load_sft_train_examples(
        "evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl",
        max_samples=8,
    )

    assert len(examples) == 8
    assert stats.sample_count == 8
    assert stats.format_parse_rate == 1.0
    assert set(stats.action_counts)
    assert all(example.target_json.startswith("{") for example in examples)
    assert all("Planner（规划器）" in example.prompt for example in examples)


def test_prompt_builder_normalizes_sample_context():
    examples, _ = load_sft_train_examples(
        "evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl",
        max_samples=1,
    )

    prompt = build_planner_prompt(examples[0].input_context)

    assert prompt.payload["original_query"]
    assert "refuse" in prompt.payload["allowed_actions"]
    assert prompt.context_key == examples[0].context_key


def test_sft_dataset_rejects_target_action_not_in_allowed_actions(tmp_path):
    bad_path = tmp_path / "bad.jsonl"
    sample = {
        "sample_id": "bad-1",
        "source_case_id": "case-1",
        "source_trace_id": "trace-1",
        "split": "train",
        "turn_index": 1,
        "input_context": {
            "query": "需要查官网吗？",
            "current_query": "需要查官网吗？",
            "subject_ids": ["subject-1"],
            "query_identifiers": {},
            "web_search_allowed": False,
            "planner_step": 0,
            "allowed_actions": ["local_search", "refuse"],
            "action_history": [],
            "latest_observation": None,
        },
        "target_decision": {
            "action": "web_search",
            "query": "需要查官网吗？",
            "reason_code": "realtime_query",
        },
        "reward_summary": {"reward_version": "reward-v1.1"},
        "gold_origin": "route_seed_gold",
        "label_source": "manual_route_seed",
        "review_status": "reviewed",
        "artifact_status": "approved_training_seed",
    }
    bad_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target_decision 不合法"):
        load_sft_train_examples(bad_path)
