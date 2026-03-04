"""
Unit tests for the News Bias Classifier web app.
These tests mock the model so they run fast in CI (no GPU/weights needed).
"""
import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import sys, os

# ── Patch heavy imports BEFORE importing app ──────────────────────────────────
# This lets CI run without downloading the 1.3GB model
mock_tokenizer = MagicMock()
mock_tokenizer.return_value = {
    "input_ids":      MagicMock(to=lambda d: MagicMock()),
    "attention_mask": MagicMock(to=lambda d: MagicMock()),
}

mock_model = MagicMock()
mock_logits = MagicMock()
mock_logits.logits = MagicMock()

import torch

@pytest.fixture
def client():
    """Create a Flask test client with a mocked model."""
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_mod, \
         patch("torch.cuda.is_available", return_value=False):

        # Mock tokenizer output
        tok_instance = MagicMock()
        tok_instance.return_value = {
            "input_ids":      torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        mock_tok.return_value = tok_instance

        # Mock model output — predict "Center" (label 1) with high confidence
        mod_instance = MagicMock()
        mod_instance.parameters.return_value = iter([torch.zeros(1)])
        output = MagicMock()
        output.logits = torch.tensor([[0.1, 2.5, 0.2]])   # Center wins
        mod_instance.return_value = output
        mod_instance.eval.return_value = None
        mock_mod.return_value = mod_instance

        # Import app after mocks are in place
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import importlib
        if "webapp.app" in sys.modules:
            del sys.modules["webapp.app"]
        if "app" in sys.modules:
            del sys.modules["app"]

        spec = importlib.util.spec_from_file_location(
            "app",
            os.path.join(os.path.dirname(__file__), "..", "webapp", "app.py")
        )
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)

        app_module.app.config["TESTING"] = True
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
