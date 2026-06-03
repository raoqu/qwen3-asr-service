"""app/api/schemas.py pydantic 模型测试（字段/默认值/校验）。

注意：这是对当前稳定结构的基线。T01 将为 HealthResponse 新增 mode/capabilities
（只增字段、向后兼容），届时更新本文件。
"""
import pytest
from pydantic import ValidationError

from app.api import schemas


def test_asr_response():
    assert schemas.ASRResponse(task_id="t1").task_id == "t1"


def test_task_status_defaults():
    r = schemas.TaskStatusResponse(task_id="t", status="pending", progress=0.0)
    assert r.result is None
    assert r.error is None


def test_task_status_progress_int_coerced_to_float():
    r = schemas.TaskStatusResponse(task_id="t", status="completed", progress=1)
    assert r.progress == 1.0
    assert isinstance(r.progress, float)


def test_task_status_missing_required_raises():
    with pytest.raises(ValidationError):
        schemas.TaskStatusResponse(task_id="t", status="pending")  # 缺 progress


def test_task_list_item_defaults():
    item = schemas.TaskListItem(
        task_id="t", status="completed", progress=1.0, created_at="2026-06-03T10:00:00",
    )
    assert item.language is None
    assert item.finished_at is None
    assert item.error is None


def test_task_list_response_coerces_dicts():
    resp = schemas.TaskListResponse(
        total=1,
        tasks=[{"task_id": "t", "status": "pending", "progress": 0.0, "created_at": "2026-06-03T10:00:00"}],
    )
    assert resp.total == 1
    assert isinstance(resp.tasks[0], schemas.TaskListItem)
    assert resp.tasks[0].task_id == "t"


def test_cancel_response():
    r = schemas.CancelResponse(task_id="t", status="cancelled", message="任务已取消")
    assert r.status == "cancelled"


def test_health_response_full():
    r = schemas.HealthResponse(
        status="ready",
        device="cpu",
        model_size="0.6b",
        align_enabled=False,
        punc_enabled=True,
        asr_backend="qwen_asr",
        vad_backend="pytorch",
        punc_backend="pytorch",
    )
    assert r.status == "ready"
    assert r.device == "cpu"


def test_health_response_missing_required_raises():
    with pytest.raises(ValidationError):
        schemas.HealthResponse(status="ready")  # 缺多个必填字段
