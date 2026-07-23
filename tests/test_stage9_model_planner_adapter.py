import json
from pathlib import Path

import pytest

from app.rag.query.contracts import (
    PlannerContext,
    PlannerReasonCode,
    QueryAction,
    SubjectResolutionStatus,
)
from app.rag.query.model_planner import ModelPlanner, ModelPlannerOutputError
from evaluation.stage9.model_planner.checkpoint_io import Stage9SftTrainingConfig
from evaluation.stage9.model_planner.sft_dataset import load_sft_train_examples
from evaluation.stage9.model_planner.sft_train import run_sft_training


def _context() -> PlannerContext:
    return PlannerContext(
        original_query="HAK 180 E020 如何处理？",
        current_query="HAK 180 E020 如何处理？",
        subject_resolution_status=SubjectResolutionStatus.CONFIRMED,
        subject_ids=["subject_hak_180"],
        query_identifiers={"alarm_code": ["E020"]},
        web_search_allowed=False,
        planner_step=0,
        max_steps=4,
        allowed_actions=[QueryAction.LOCAL_SEARCH, QueryAction.REFUSE],
    )


def test_model_planner_uses_generator_and_decodes_decision():
    planner = ModelPlanner(
        policy_version="test-model-planner",
        generate_text=lambda **_: (
            '{"action":"local_search","query":"HAK 180 E020 如何处理？",'
            '"reason_code":"initial_local_search"}'
        ),
    )

    decision = planner.plan(_context())

    assert decision.action == QueryAction.LOCAL_SEARCH
    assert decision.reason_code == PlannerReasonCode.INITIAL_LOCAL_SEARCH


def test_model_planner_rejects_invalid_generator_output():
    planner = ModelPlanner(
        policy_version="test-model-planner",
        generate_text=lambda **_: '{"action":"web_search","query":"q","reason_code":"realtime_query"}',
    )

    with pytest.raises(ModelPlannerOutputError, match="action_not_allowed"):
        planner.plan(_context())


def test_debug_smoke_checkpoint_loads_through_model_planner(tmp_path):
    train_data = "evaluation/stage9/artifacts/sft/sft_planner_stage9_train.jsonl"
    train_manifest = "evaluation/stage9/artifacts/sft/sft_planner_stage9_manifest.json"
    reward_profile = "evaluation/stage9/configs/reward_v1_1_training_profile.json"
    config = Stage9SftTrainingConfig(
        run_name="planner-sft-stage9-test-smoke",
        training_backend="debug_memorized",
        base_model_id="stage9-debug-memorized",
        train_data=train_data,
        train_manifest=train_manifest,
        reward_profile=reward_profile,
        snapshot_id="stage85-env-20260721-v2",
        output_root=str(tmp_path),
        num_epochs=0,
        max_train_samples=2,
        save_training_preview_count=1,
    )
    manifest = run_sft_training(config)
    checkpoint_dir = tmp_path / manifest.run_id
    examples, _ = load_sft_train_examples(train_data, max_samples=1)
    context_payload = examples[0].input_context

    planner = ModelPlanner.from_checkpoint(checkpoint_dir)
    decision = planner.plan(PlannerContext(
        original_query=context_payload["original_query"],
        current_query=context_payload["current_query"],
        subject_resolution_status=SubjectResolutionStatus.CONFIRMED,
        subject_ids=context_payload["subject_ids"],
        query_identifiers=context_payload["query_identifiers"],
        web_search_allowed=context_payload["web_search_allowed"],
        planner_step=context_payload["planner_step"],
        max_steps=context_payload["max_steps"],
        allowed_actions=context_payload["allowed_actions"],
    ))

    assert decision.model_dump(mode="json") == examples[0].target_decision
    metrics = json.loads((checkpoint_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["format_parse_rate"] == 1.0


def test_model_planner_runtime_does_not_import_stage9_runtime():
    runtime_dir = Path("app/rag/query/model_planner")
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in runtime_dir.glob("*.py")
    )

    assert "evaluation.stage9.model_planner" not in sources
