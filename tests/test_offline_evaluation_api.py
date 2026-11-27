import pytest

from src.api import main
from src.models.offline_evaluation import get_offline_evaluation


def test_offline_evaluation_reads_saved_artifacts():
    result = get_offline_evaluation()

    assert result["status"] == "available"
    assert result["mode"] == "offline_evaluation"
    assert result["live_inference"] is False
    assert result["features"] == 47
    assert result["sequence_length"] == 30
    assert result["artifact_available"] is True
    assert 0 < result["metrics"]["f1_score"] <= 1


def test_offline_evaluation_endpoint_preserves_read_only_contract():
    result = main.offline_evaluation()

    assert result["live_inference"] is False
    assert result["metrics"]["true_positive"] == 18615


def test_offline_evaluation_endpoint_fails_closed(monkeypatch):
    def broken_reader():
        raise ValueError("artifact validation failed")

    monkeypatch.setattr(main, "get_offline_evaluation", broken_reader)

    with pytest.raises(main.HTTPException) as error:
        main.offline_evaluation()

    assert error.value.status_code == 503
    assert "artifact validation failed" in error.value.detail
