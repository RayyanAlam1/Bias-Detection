"""
Unit tests for the News Bias Classifier web app.
Mocks ALL heavy operations (safetensors load, tokenizer, model)
so tests run in CI with no GPU, no model weights, no disk access.
"""
import json
import os
import sys
import importlib
import importlib.util
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch


# ─── Helpers ──────────────────────────────────────────────────────────────────

WEBAPP_DIR = os.path.join(os.path.dirname(__file__), "..", "webapp")
TEMPLATES_DIR = os.path.join(WEBAPP_DIR, "templates")
APP_PY = os.path.join(WEBAPP_DIR, "app.py")


def _make_mock_tokenizer():
    """Tokenizer that returns fixed tensors — no vocab/files needed."""
    tok = MagicMock()
    tok.return_value = {
        "input_ids":      torch.zeros(1, 512, dtype=torch.long),
        "attention_mask": torch.ones(1, 512, dtype=torch.long),
    }
    return tok


def _make_mock_model():
    """Model whose forward() returns logits predicting Center (label 1)."""
    mod = MagicMock()
    output = MagicMock()
    output.logits = torch.tensor([[0.1, 2.5, 0.2]])   # Center wins
    mod.return_value = output
    mod.eval.return_value = None
    mod.parameters.return_value = iter([torch.zeros(1)])
    mod.load_state_dict = MagicMock()
    mod.to = MagicMock(return_value=mod)
    mod.half = MagicMock(return_value=mod)
    return mod


def _load_app_module():
    """
    Load webapp/app.py fresh with all heavy deps mocked.
    Patches are applied BEFORE module-level code runs so
    safetensors / transformers calls never execute.
    """
    for key in list(sys.modules.keys()):
        if key in ("app", "webapp.app"):
            del sys.modules[key]

    mock_tok = _make_mock_tokenizer()
    mock_mod = _make_mock_model()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tok), \
         patch("transformers.AutoConfig.from_pretrained", return_value=MagicMock()), \
         patch("transformers.AutoModelForSequenceClassification.from_config",
               return_value=mock_mod), \
         patch("safetensors.torch.load_file", return_value={}), \
         patch("torch.cuda.is_available", return_value=False):

        spec = importlib.util.spec_from_file_location("app", APP_PY)
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)

    return app_module


# ─── Fixture ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Flask test client with mocked model — no weights needed."""
    app_module = _load_app_module()
    app_module.app.config["TESTING"] = True
    app_module.app.template_folder = TEMPLATES_DIR   # fix TemplateNotFound in CI
    with app_module.app.test_client() as c:
        yield c


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_homepage_returns_200(client):
    """GET / should return the HTML page."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Bias" in resp.data or b"bias" in resp.data or b"html" in resp.data.lower()


def test_predict_returns_json(client):
    """POST /predict with valid text should return JSON with required keys."""
    resp = client.post(
        "/predict",
        data=json.dumps({"text": "The government announced new economic policies today."}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "label" in data
    assert "name" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "reasoning" in data
    assert "summary" in data["reasoning"]
    assert data["name"] in ("Left", "Center", "Right")
    assert 0 <= data["confidence"] <= 100


def test_predict_empty_text_returns_400(client):
    """POST /predict with empty text should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({"text": ""}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_predict_missing_text_returns_400(client):
    """POST /predict with no text key should return 400."""
    resp = client.post(
        "/predict",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_predict_text_endpoint_returns_json(client):
    """POST /predict_text should mirror /predict schema."""
    resp = client.post(
        "/predict_text",
        data=json.dumps({"text": "A neutral policy update was released."}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sentence_analysis" in data
    assert "sentence_count" in data


def test_feedback_endpoint_accepts_agree(client):
    """POST /feedback should persist lightweight feedback metadata."""
    resp = client.post(
        "/feedback",
        data=json.dumps(
            {
                "feedback": "agree",
                "source_type": "text",
                "source_value": "sample text",
                "predicted_label": "Center",
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json().get("ok") is True


def test_feedback_endpoint_rejects_bad_value(client):
    """POST /feedback should validate allowed values."""
    resp = client.post(
        "/feedback",
        data=json.dumps(
            {
                "feedback": "maybe",
                "source_type": "text",
                "source_value": "sample text",
                "predicted_label": "Center",
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_probabilities_sum_to_100(client):
    """All 3 class probabilities should sum to ~100%."""
    resp = client.post(
        "/predict",
        data=json.dumps({"text": "Breaking news: elections scheduled for next month."}),
        content_type="application/json",
    )
    data = resp.get_json()
    total = sum(data["probabilities"].values())
    assert abs(total - 100.0) < 1.0, f"Probabilities sum to {total}, expected ~100"


def test_label_info_valid(client):
    """Returned label index should match the returned name."""
    resp = client.post(
        "/predict",
        data=json.dumps({"text": "Senator calls for bipartisan cooperation."}),
        content_type="application/json",
    )
    data = resp.get_json()
    label_map = {0: "Left", 1: "Center", 2: "Right"}
    assert data["name"] == label_map[data["label"]]
