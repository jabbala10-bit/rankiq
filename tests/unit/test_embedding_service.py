"""Unit tests for src/services/embedding/embedding_service.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.exceptions import EmbeddingError, ModelNotLoadedError
from src.services.embedding.embedding_service import EmbeddingService


class FakeEncodeResult(list):
    """Mimics numpy array's .tolist() interface for a single embedding."""

    def tolist(self):
        return list(self)


class TestLoadModel:
    def test_load_model_succeeds(self, test_settings):
        service = EmbeddingService(test_settings)
        with patch("sentence_transformers.SentenceTransformer") as MockModel:
            MockModel.return_value = MagicMock()
            service.load_model()
        assert service.is_loaded is True

    def test_load_model_idempotent(self, test_settings):
        service = EmbeddingService(test_settings)
        with patch("sentence_transformers.SentenceTransformer") as MockModel:
            MockModel.return_value = MagicMock()
            service.load_model()
            service.load_model()
            assert MockModel.call_count == 1

    def test_load_model_failure_raises_embedding_error(self, test_settings):
        service = EmbeddingService(test_settings)
        with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("boom")):
            with pytest.raises(EmbeddingError):
                service.load_model()


class TestEmbedText:
    def test_raises_if_model_not_loaded(self, test_settings):
        service = EmbeddingService(test_settings)
        with pytest.raises(ModelNotLoadedError):
            service.embed_text("id1", "some text")

    def test_embed_text_returns_product_embedding(self, test_settings):
        service = EmbeddingService(test_settings)
        fake_model = MagicMock()
        fake_model.encode.return_value = FakeEncodeResult([0.1, 0.2, 0.3])
        service._model = fake_model

        result = service.embed_text("id1", "waterproof jacket")
        assert result.product_id == "id1"
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.dimensions == 3

    def test_embed_text_failure_raises_embedding_error(self, test_settings):
        service = EmbeddingService(test_settings)
        fake_model = MagicMock()
        fake_model.encode.side_effect = RuntimeError("encode failed")
        service._model = fake_model

        with pytest.raises(EmbeddingError):
            service.embed_text("id1", "text")


class TestEmbedProduct:
    def test_embed_product_uses_searchable_text(self, test_settings, sample_product):
        service = EmbeddingService(test_settings)
        fake_model = MagicMock()
        fake_model.encode.return_value = FakeEncodeResult([0.1, 0.2])
        service._model = fake_model

        result = service.embed_product(sample_product)
        assert result.product_id == sample_product.product_id

        call_args = fake_model.encode.call_args[0]
        assert sample_product.title in call_args[0]


class TestEmbedBatch:
    def test_raises_if_model_not_loaded(self, test_settings, sample_catalog):
        service = EmbeddingService(test_settings)
        with pytest.raises(ModelNotLoadedError):
            service.embed_batch(sample_catalog)

    def test_empty_batch_returns_empty_list(self, test_settings):
        service = EmbeddingService(test_settings)
        service._model = MagicMock()
        assert service.embed_batch([]) == []

    def test_batch_returns_one_embedding_per_product(self, test_settings, sample_catalog):
        import numpy as np

        service = EmbeddingService(test_settings)
        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.rand(len(sample_catalog), 8)
        service._model = fake_model

        results = service.embed_batch(sample_catalog)
        assert len(results) == len(sample_catalog)
        for product, embedding in zip(sample_catalog, results):
            assert embedding.product_id == product.product_id
            assert embedding.dimensions == 8
