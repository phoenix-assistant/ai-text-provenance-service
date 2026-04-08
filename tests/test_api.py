"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from ai_text_provenance.api.app import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Test the health endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns OK."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data


class TestClassifyEndpoint:
    """Test the classify endpoint."""

    def test_classify_valid_text(self, client):
        """Test classification of valid text."""
        response = client.post(
            "/classify",
            json={
                "text": "This is a test sentence that is long enough for classification. It needs to be at least 50 characters.",
                "include_features": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert data["prediction"] in ["human", "ai", "polished_human", "humanized_ai"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "probabilities" in data
        assert len(data["probabilities"]) == 4

    def test_classify_with_features(self, client):
        """Test classification with feature extraction."""
        response = client.post(
            "/classify",
            json={
                "text": "This is a longer test sentence that includes multiple clauses, because we want to test feature extraction properly with enough content.",
                "include_features": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "features" in data
        assert data["features"] is not None
        assert "rst" in data["features"]
        assert "linguistic" in data["features"]
        assert "statistical" in data["features"]

    def test_classify_text_too_short(self, client):
        """Test rejection of too-short text."""
        response = client.post(
            "/classify",
            json={"text": "Too short"},
        )

        assert response.status_code == 422  # Validation error

    def test_classify_missing_text(self, client):
        """Test rejection of missing text field."""
        response = client.post(
            "/classify",
            json={},
        )

        assert response.status_code == 422


class TestBatchEndpoint:
    """Test the batch classify endpoint."""

    def test_batch_classify(self, client):
        """Test batch classification."""
        texts = [
            "First text for batch processing that is long enough to meet the minimum character requirement for validation.",
            "Second text for batch processing that also meets the minimum character count for proper classification testing.",
        ]

        response = client.post(
            "/classify/batch",
            json={
                "texts": texts,
                "include_features": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert len(data["results"]) == 2
        assert "total" in data
        assert data["total"] == 2
        assert "processing_time_ms" in data

    def test_batch_empty_list(self, client):
        """Test rejection of empty batch."""
        response = client.post(
            "/classify/batch",
            json={"texts": []},
        )

        assert response.status_code == 422

    def test_batch_too_large(self, client):
        """Test rejection of oversized batch."""
        texts = ["Sample text that is long enough for classification testing purposes."] * 101

        response = client.post(
            "/classify/batch",
            json={"texts": texts},
        )

        assert response.status_code == 400


class TestRootEndpoint:
    """Test the root endpoint."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "service" in data
        assert "version" in data
        assert "docs" in data
