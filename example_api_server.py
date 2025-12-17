"""
Example Embedding API Server
=============================
This is an example server that shows the expected API interface for eval_api.py

Usage:
    # Install dependencies
    pip install fastapi uvicorn

    # Run server
    uvicorn example_api_server:app --host 0.0.0.0 --port 8000

The server exposes:
    POST /embed - Get embeddings for text/image inputs
    GET /health - Health check endpoint
"""

import base64
import io
import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VLM Embedding API", version="1.0.0")


# ============================================================================
# Request/Response Models
# ============================================================================
class EmbedRequest(BaseModel):
    """Request model for embedding endpoint"""
    texts: List[str]
    images: Optional[List[Optional[str]]] = None  # Base64 encoded images
    is_query: bool = True

    class Config:
        schema_extra = {
            "example": {
                "texts": ["A photo of a cat", "A photo of a dog"],
                "images": [None, None],  # or base64 encoded strings
                "is_query": True
            }
        }


class EmbedResponse(BaseModel):
    """Response model for embedding endpoint"""
    embeddings: List[List[float]]
    dimensions: int

    class Config:
        schema_extra = {
            "example": {
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                "dimensions": 3
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool


# ============================================================================
# Global Model (Replace with your actual model)
# ============================================================================
class DummyEmbeddingModel:
    """
    Dummy embedding model for demonstration.
    Replace this with your actual VLM model.
    """

    def __init__(self, embedding_dim: int = 1024):
        self.embedding_dim = embedding_dim
        self.is_loaded = True
        logger.info(f"Dummy model initialized with embedding_dim={embedding_dim}")

    def encode(
        self,
        texts: List[str],
        images: List[Optional[Image.Image]] = None,
        is_query: bool = True
    ) -> np.ndarray:
        """
        Encode text and images into embeddings.

        Args:
            texts: List of text inputs
            images: List of PIL Images (None for text-only inputs)
            is_query: Whether this is a query (vs. candidate)

        Returns:
            numpy array of shape [batch_size, embedding_dim]
        """
        batch_size = len(texts)

        # TODO: Replace this with your actual model inference
        # This is just a dummy implementation that returns random embeddings

        embeddings = []
        for i, text in enumerate(texts):
            # Create a deterministic embedding based on text hash
            # This ensures the same input produces the same output
            np.random.seed(hash(text) % (2**32))
            emb = np.random.randn(self.embedding_dim).astype(np.float32)

            # If image is provided, modify the embedding slightly
            if images and images[i] is not None:
                # Add image influence (in practice, use actual image encoding)
                img_size = images[i].size
                np.random.seed((hash(text) + img_size[0] * img_size[1]) % (2**32))
                img_emb = np.random.randn(self.embedding_dim).astype(np.float32)
                emb = (emb + img_emb) / 2

            # Normalize to unit length
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)

        return np.stack(embeddings)


# Initialize model
model = DummyEmbeddingModel(embedding_dim=1024)


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model.is_loaded
    )


@app.post("/embed", response_model=EmbedResponse)
async def get_embeddings(request: EmbedRequest):
    """
    Get embeddings for text and optional image inputs.

    - **texts**: List of text strings
    - **images**: Optional list of base64-encoded images (same length as texts, use null for text-only)
    - **is_query**: Whether these are query inputs (affects some models' behavior)
    """
    try:
        # Validate input
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts cannot be empty")

        # Decode images if provided
        images = None
        if request.images:
            if len(request.images) != len(request.texts):
                raise HTTPException(
                    status_code=400,
                    detail=f"images length ({len(request.images)}) must match texts length ({len(request.texts)})"
                )

            images = []
            for img_b64 in request.images:
                if img_b64 is None:
                    images.append(None)
                else:
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        img = Image.open(io.BytesIO(img_bytes))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        images.append(img)
                    except Exception as e:
                        logger.error(f"Failed to decode image: {e}")
                        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

        # Get embeddings
        embeddings = model.encode(
            texts=request.texts,
            images=images,
            is_query=request.is_query
        )

        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dimensions=embeddings.shape[1]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Example of how to integrate your actual model
# ============================================================================
"""
# Example: Integrating Qwen2-VL model

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

class Qwen2VLEmbeddingModel:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        self.is_loaded = True
    
    def encode(self, texts, images=None, is_query=True):
        with torch.no_grad():
            # Process inputs
            if images:
                inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
            else:
                inputs = self.processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
            
            # Get hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            
            # Pool: take last token
            attention_mask = inputs.attention_mask
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[torch.arange(batch_size), seq_lens]
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
            return embeddings.cpu().float().numpy()

# Replace the dummy model:
# model = Qwen2VLEmbeddingModel()
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

