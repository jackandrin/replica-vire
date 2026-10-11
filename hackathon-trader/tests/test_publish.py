"""publish_models refuses to pair metrics and models from different training runs (checked before any DB call)."""
import json

import pytest

from src.publish_models import publish


def test_rejects_mismatched_run(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"version": "2026-10-01T00:00:00"}))
    (tmp_path / "metrics.json").write_text(json.dumps({"meta": {"trained_at": "2026-09-01T00:00:00"}}))
    with pytest.raises(RuntimeError, match="different run"):
        publish(tmp_path, tmp_path)
