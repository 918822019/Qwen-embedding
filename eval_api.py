"""
VLM2Vec API-based Evaluation Script
====================================
This script evaluates VLM models via ByteDance Ark API endpoints.

Usage:
    # Evaluate all modalities (image, video, visdoc) at once
    python eval_api.py \
        --api_key "$ARK_API_KEY" \
        --model "ep-20250917160742-f9hzv" \
        --eval_all_modalities \
        --encode_output_path output/api_eval/ \
        --data_basedir data/vlm2vec_eval

    # Evaluate specific modality
    python eval_api.py \
        --api_key "$ARK_API_KEY" \
        --model "ep-20250917160742-f9hzv" \
        --dataset_config experiments/public/eval/image.yaml \
        --encode_output_path output/api_eval/ \
        --data_basedir data/vlm2vec_eval

    # Evaluate single dataset
    python eval_api.py \
        --api_key "$ARK_API_KEY" \
        --model "ep-20250917160742-f9hzv" \
        --dataset_config experiments/public/eval/image.yaml \
        --dataset_name ImageNet-1K \
        --encode_output_path output/api_eval/

ByteDance Ark API Format:
    POST /api/v3/embeddings/multimodal
    Headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer $ARK_API_KEY"
    }
    Request Body: {
        "model": "ep-xxx",
        "input": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }
    Response: {
        "data": [{"embedding": [0.1, 0.2, ...]}]
    }
"""

import argparse
import base64
import datetime
import io
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
import yaml
from PIL import Image
from tqdm import tqdm

# Import from existing VLM2Vec codebase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.eval_utils.metrics import RankingMetrics

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# All modality config files
ALL_MODALITY_CONFIGS = {
    "image": "experiments/public/eval/image.yaml",
    "video": "experiments/public/eval/video.yaml",
    "visdoc": "experiments/public/eval/visdoc.yaml",
}


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class APIConfig:
    """Configuration for ByteDance Ark API-based evaluation"""
    api_base_url: str = "https://ark-cn-beijing.bytedance.net/api/v3/embeddings/multimodal"
    api_key: str = ""  # ARK_API_KEY
    model: str = "ep-20250917160742-f9hzv"  # Ark model endpoint ID
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 1  # Ark API processes one input at a time
    num_workers: int = 4


@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    dataset_config: str = None
    dataset_configs: List[str] = None  # Multiple config files for all modalities
    data_basedir: str = None
    encode_output_path: str = "./output/api_eval"
    image_resolution: Tuple[int, int] = (672, 672)
    skip_existing: bool = True
    eval_all_modalities: bool = False


# ============================================================================
# API Client
# ============================================================================
class EmbeddingAPIClient:
    """Client for calling ByteDance Ark Multimodal Embedding API"""

    def __init__(self, config: APIConfig):
        self.config = config
        self.api_url = config.api_base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}"
        })

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL"""
        buffered = io.BytesIO()
        # Save as JPEG for smaller size, or PNG for quality
        image_format = "JPEG"
        if image.mode == "RGBA":
            image_format = "PNG"
        image.save(buffered, format=image_format)
        b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        mime_type = "image/jpeg" if image_format == "JPEG" else "image/png"
        return f"data:{mime_type};base64,{b64_str}"

    def _load_image(self, image_path: str, resolution: Tuple[int, int] = None) -> Image.Image:
        """Load and optionally resize image"""
        image = Image.open(image_path)
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')
        if resolution:
            image = image.resize(resolution)
        return image

    def _build_input(self, text: str, image_path: Optional[str], resolution: Tuple[int, int] = None) -> List[Dict]:
        """Build input array for Ark API"""
        input_items = []

        # Add image first if exists (following the example format)
        if image_path and os.path.exists(image_path):
            image = self._load_image(image_path, resolution)
            image_url = self._image_to_base64_url(image)
            input_items.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })

        # Add text
        if text and text.strip():
            input_items.append({
                "type": "text",
                "text": text
            })

        return input_items

    def get_embedding_single(
        self,
        text: str,
        image_path: Optional[str] = None,
        resolution: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Get embedding for a single text/image input from Ark API

        Args:
            text: Text input
            image_path: Optional image path
            resolution: Optional image resolution for resizing

        Returns:
            numpy array of embedding [embedding_dim]
        """
        # Build request
        input_items = self._build_input(text, image_path, resolution)

        if not input_items:
            raise ValueError("At least text or image must be provided")

        request_data = {
            "model": self.config.model,
            "input": input_items
        }

        # Make request with retries
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    json=request_data,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                result = response.json()

                # Parse Ark API response format
                # Response: {"data": [{"embedding": [0.1, 0.2, ...]}]}
                embedding = np.array(result["data"][0]["embedding"], dtype=np.float32)
                return embedding

            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"API request failed after {self.config.max_retries} attempts: {e}")
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse API response: {e}")
                logger.error(f"Response: {response.text if 'response' in dir() else 'N/A'}")
                raise

        return None

    def get_embeddings(
        self,
        texts: List[str],
        image_paths: List[Optional[str]] = None,
        is_query: bool = True,
        resolution: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        Get embeddings for multiple inputs (calls API sequentially)

        Args:
            texts: List of text inputs
            image_paths: List of image paths (None for text-only)
            is_query: Whether these are query inputs (not used in Ark API)
            resolution: Optional image resolution for resizing

        Returns:
            numpy array of embeddings [batch_size, embedding_dim]
        """
        embeddings = []

        if image_paths is None:
            image_paths = [None] * len(texts)

        for text, image_path in zip(texts, image_paths):
            emb = self.get_embedding_single(text, image_path, resolution)
            embeddings.append(emb)

        return np.stack(embeddings)

    def health_check(self) -> bool:
        """Check if API is available with a simple test request"""
        try:
            test_request = {
                "model": self.config.model,
                "input": [{"type": "text", "text": "test"}]
            }
            response = self.session.post(
                self.api_url,
                json=test_request,
                timeout=30
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False


# ============================================================================
# Dataset Loaders (Simplified versions for API evaluation)
# ============================================================================
class SimpleDatasetLoader:
    """Simplified dataset loader for API evaluation"""

    DATASET_HF_PATHS = {
        # Image Classification datasets
        "ImageNet-1K": "ziyjiang/MMEB_Test_Instruct",
        "N24News": "ziyjiang/MMEB_Test_Instruct",
        "HatefulMemes": "ziyjiang/MMEB_Test_Instruct",
        "VOC2007": "ziyjiang/MMEB_Test_Instruct",
        "SUN397": "ziyjiang/MMEB_Test_Instruct",
        "Place365": "ziyjiang/MMEB_Test_Instruct",
        "ImageNet-A": "ziyjiang/MMEB_Test_Instruct",
        "ImageNet-R": "ziyjiang/MMEB_Test_Instruct",
        "ObjectNet": "ziyjiang/MMEB_Test_Instruct",
        "Country211": "ziyjiang/MMEB_Test_Instruct",
    }

    @staticmethod
    def load_image_cls_dataset(dataset_name: str, image_root: str):
        """Load image classification dataset"""
        try:
            from datasets import load_dataset
            dataset = load_dataset("ziyjiang/MMEB_Test_Instruct", dataset_name, split="test")

            queries = []
            candidates = []
            gt_infos = []
            all_labels = set()

            for row in dataset:
                # Query: image with instruction
                query_text = row['qry_inst'].replace("<|image_1|>", "") + "\n" + row['qry_text'] + "\n"
                query_image = os.path.join(image_root, row['qry_img_path'])
                queries.append({
                    "text": query_text,
                    "image_path": query_image
                })

                # Ground truth info
                gt_infos.append({
                    "cand_names": row['tgt_text'],
                    "label_name": row['tgt_text'][0]  # First one is the correct label
                })
                all_labels.update(row['tgt_text'])

            # Candidates: all unique labels (text only)
            for label in all_labels:
                candidates.append({
                    "text": label,
                    "image_path": None,
                    "cand_name": label
                })

            return queries, candidates, gt_infos

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    @staticmethod
    def load_generic_dataset(task_config: dict, data_basedir: str = None):
        """Load dataset based on task config"""
        dataset_parser = task_config.get("dataset_parser", "")
        dataset_name = task_config.get("dataset_name", "")

        # Construct paths
        image_root = task_config.get("image_root", "")
        if data_basedir and image_root:
            image_root = os.path.join(data_basedir, image_root)

        if dataset_parser == "image_cls":
            return SimpleDatasetLoader.load_image_cls_dataset(
                dataset_name=dataset_name,
                image_root=image_root
            )
        else:
            # For other dataset types, try to use the original loaders
            logger.warning(f"Dataset parser '{dataset_parser}' not fully implemented for API eval. "
                          f"Falling back to basic loading.")
            return SimpleDatasetLoader._load_from_original(task_config, data_basedir)

    @staticmethod
    def _load_from_original(task_config: dict, data_basedir: str = None):
        """Fallback to original dataset loading mechanism"""
        try:
            from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
            from src.arguments import ModelArguments, DataArguments

            # Create minimal args
            model_args = ModelArguments(model_name="placeholder", model_backbone="qwen2_vl")
            data_args = DataArguments(image_resolution="mid")

            # Update paths
            if data_basedir:
                for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                    if task_config.get(key):
                        task_config[key] = os.path.join(data_basedir, task_config[key])

            # Load using original loader
            eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(
                model_args=model_args, data_args=data_args, **task_config
            )
            eval_cand_dataset = generate_cand_dataset(eval_qry_dataset, corpus)

            # Convert to simple format
            queries = []
            gt_infos = []
            for row in eval_qry_dataset:
                query_text = row['query_text'][0] if row['query_text'] else ""
                query_image = None
                if row['query_image'] and row['query_image'][0]:
                    paths = row['query_image'][0].get('paths', [])
                    if paths and paths[0]:
                        query_image = paths[0]

                queries.append({"text": query_text, "image_path": query_image})
                gt_infos.append(row['dataset_infos'])

            candidates = []
            for row in eval_cand_dataset:
                cand_text = row['cand_text'][0] if row['cand_text'] else ""
                cand_image = None
                if row['cand_image'] and row['cand_image'][0]:
                    paths = row['cand_image'][0].get('paths', [])
                    if paths and paths[0]:
                        cand_image = paths[0]

                candidates.append({
                    "text": cand_text,
                    "image_path": cand_image,
                    "cand_name": row['dataset_infos']['cand_name']
                })

            return queries, candidates, gt_infos

        except Exception as e:
            logger.error(f"Failed to load dataset using original loader: {e}")
            raise


# ============================================================================
# Main Evaluation Logic
# ============================================================================
class APIEvaluator:
    """Main evaluator using API for embeddings"""

    def __init__(self, api_config: APIConfig, eval_config: EvalConfig):
        self.api_config = api_config
        self.eval_config = eval_config
        self.client = EmbeddingAPIClient(api_config)
        self.dataset_loader = SimpleDatasetLoader()

        os.makedirs(eval_config.encode_output_path, exist_ok=True)

    def encode_batch(
        self,
        items: List[Dict],
        is_query: bool,
        desc: str = "Encoding"
    ) -> Tuple[np.ndarray, List[str]]:
        """Encode a batch of items"""
        all_embeddings = []
        all_keys = []

        batch_size = self.api_config.batch_size

        for i in tqdm(range(0, len(items), batch_size), desc=desc):
            batch = items[i:i + batch_size]

            texts = [item.get("text", "") for item in batch]
            image_paths = [item.get("image_path") for item in batch]

            # Check if any images exist
            has_images = any(p and os.path.exists(p) for p in image_paths)

            embeddings = self.client.get_embeddings(
                texts=texts,
                image_paths=image_paths if has_images else None,
                is_query=is_query,
                resolution=self.eval_config.image_resolution
            )

            all_embeddings.append(embeddings)

            # Collect keys
            for item in batch:
                key = item.get("cand_name", item.get("text", str(len(all_keys))))
                all_keys.append(key)

        return np.vstack(all_embeddings), all_keys

    def evaluate_dataset(self, dataset_name: str, task_config: dict) -> Dict[str, float]:
        """Evaluate a single dataset"""
        logger.info(f"--- Evaluating {dataset_name} ---")

        # Paths for caching
        query_embed_path = os.path.join(self.eval_config.encode_output_path, f"{dataset_name}_qry.pkl")
        cand_embed_path = os.path.join(self.eval_config.encode_output_path, f"{dataset_name}_tgt.pkl")
        info_path = os.path.join(self.eval_config.encode_output_path, f"{dataset_name}_info.jsonl")
        score_path = os.path.join(self.eval_config.encode_output_path, f"{dataset_name}_score.json")

        # Check if already evaluated
        if self.eval_config.skip_existing and os.path.exists(score_path):
            logger.info(f"Score already exists, loading from {score_path}")
            with open(score_path, 'r') as f:
                return json.load(f)

        # Load dataset
        try:
            queries, candidates, gt_infos = self.dataset_loader.load_generic_dataset(
                task_config, self.eval_config.data_basedir
            )
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return {}

        logger.info(f"Loaded {len(queries)} queries and {len(candidates)} candidates")

        # Encode queries
        if not os.path.exists(query_embed_path):
            logger.info("Encoding queries...")
            query_embeds, _ = self.encode_batch(queries, is_query=True, desc=f"Queries for {dataset_name}")
            with open(query_embed_path, 'wb') as f:
                pickle.dump(query_embeds, f)
            with open(info_path, 'w') as f:
                for info in gt_infos:
                    f.write(json.dumps(info) + '\n')
        else:
            logger.info(f"Loading cached query embeddings from {query_embed_path}")
            with open(query_embed_path, 'rb') as f:
                query_embeds = pickle.load(f)

        # Encode candidates
        if not os.path.exists(cand_embed_path):
            logger.info("Encoding candidates...")
            cand_embeds_arr, cand_keys = self.encode_batch(candidates, is_query=False, desc=f"Candidates for {dataset_name}")
            cand_embed_dict = {key: embed for key, embed in zip(cand_keys, cand_embeds_arr)}
            with open(cand_embed_path, 'wb') as f:
                pickle.dump(cand_embed_dict, f)
        else:
            logger.info(f"Loading cached candidate embeddings from {cand_embed_path}")
            with open(cand_embed_path, 'rb') as f:
                cand_embed_dict = pickle.load(f)

        # Ensure variables are defined (for static analysis)
        assert query_embeds is not None, "Query embeddings not loaded"
        assert cand_embed_dict is not None, "Candidate embeddings not loaded"

        # Load ground truth info
        gt_infos = [json.loads(l) for l in open(info_path)]

        # Calculate scores
        logger.info("Calculating scores...")
        pred_dicts = []

        rank_against_all = task_config.get("eval_type", "global") == "global"

        if rank_against_all:
            # Global ranking
            cand_keys = list(cand_embed_dict.keys())
            cand_embeds = np.stack([cand_embed_dict[key] for key in cand_keys])

            # Normalize for cosine similarity
            query_embeds_norm = query_embeds / (np.linalg.norm(query_embeds, axis=1, keepdims=True) + 1e-8)
            cand_embeds_norm = cand_embeds / (np.linalg.norm(cand_embeds, axis=1, keepdims=True) + 1e-8)

            cosine_scores = np.dot(query_embeds_norm, cand_embeds_norm.T)
            ranked_indices = np.argsort(-cosine_scores, axis=1)

            for qid, (ranked_idx, gt_info) in enumerate(zip(ranked_indices, gt_infos)):
                rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                rel_scores = gt_info.get("rel_scores")
                pred_dicts.append({
                    "prediction": [cand_keys[i] for i in ranked_idx],
                    "label": rel_docids,
                    "rel_scores": rel_scores,
                })
        else:
            # Local ranking (within candidate subset)
            for qid, (qry_embed, gt_info) in enumerate(zip(query_embeds, gt_infos)):
                cand_names = gt_info.get("cand_names", [])
                if not cand_names:
                    continue

                cand_embeds = np.stack([cand_embed_dict[name] for name in cand_names if name in cand_embed_dict])

                # Normalize
                qry_embed_norm = qry_embed / (np.linalg.norm(qry_embed) + 1e-8)
                cand_embeds_norm = cand_embeds / (np.linalg.norm(cand_embeds, axis=1, keepdims=True) + 1e-8)

                cosine_scores = np.dot(qry_embed_norm, cand_embeds_norm.T)
                ranked_idx = np.argsort(-cosine_scores)

                rel_docids = gt_info["label_name"] if isinstance(gt_info["label_name"], list) else [gt_info["label_name"]]
                rel_scores = gt_info.get("rel_scores")

                pred_dicts.append({
                    "prediction": [cand_names[i] for i in ranked_idx],
                    "label": rel_docids,
                    "rel_scores": rel_scores,
                })

        # Calculate metrics
        metrics_to_report = task_config.get("metrics") or ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"]
        metrics = RankingMetrics(metrics_to_report)
        score_dict = metrics.evaluate(pred_dicts)

        # Add metadata
        score_dict["num_queries"] = len(queries)
        score_dict["num_candidates"] = len(candidates)
        score_dict["num_predictions"] = len(pred_dicts)

        # Save scores
        with open(score_path, 'w') as f:
            json.dump(score_dict, f, indent=2)

        # Print results
        formatted = {k: f"{v:.4f}" for k, v in score_dict.items() if isinstance(v, float)}
        logger.info(f"Results for {dataset_name}:")
        logger.info(formatted)

        return score_dict

    def evaluate_all(self, dataset_config_path: str = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all datasets in a single config file"""
        config_path = dataset_config_path or self.eval_config.dataset_config

        # Check API health
        logger.info(f"Checking API health at {self.api_config.api_base_url}...")
        if not self.client.health_check():
            logger.warning("API health check failed, but continuing anyway...")
        else:
            logger.info("API is healthy!")

        # Load dataset config
        with open(config_path, 'r') as f:
            dataset_configs = yaml.safe_load(f)

        all_scores = {}

        for dataset_name, task_config in dataset_configs.items():
            try:
                scores = self.evaluate_dataset(dataset_name, task_config)
                all_scores[dataset_name] = scores
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                all_scores[dataset_name] = {"error": str(e)}

        return all_scores

    def evaluate_all_modalities(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Evaluate all modalities (image, video, visdoc) at once"""
        logger.info("=" * 60)
        logger.info("Starting evaluation for ALL modalities: image, video, visdoc")
        logger.info("=" * 60)

        all_modality_scores = {}

        for modality, config_path in ALL_MODALITY_CONFIGS.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Evaluating modality: {modality.upper()}")
            logger.info(f"Config file: {config_path}")
            logger.info("=" * 60)

            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}, skipping {modality}")
                all_modality_scores[modality] = {"error": f"Config file not found: {config_path}"}
                continue

            # Create modality-specific output directory
            modality_output_path = os.path.join(self.eval_config.encode_output_path, modality)
            os.makedirs(modality_output_path, exist_ok=True)

            # Temporarily update output path for this modality
            original_output_path = self.eval_config.encode_output_path
            self.eval_config.encode_output_path = modality_output_path

            try:
                modality_scores = self.evaluate_all(config_path)
                all_modality_scores[modality] = modality_scores

                # Save modality-specific summary
                modality_summary_path = os.path.join(modality_output_path, "summary.json")
                with open(modality_summary_path, 'w') as f:
                    json.dump(modality_scores, f, indent=2)
                logger.info(f"Modality {modality} summary saved to {modality_summary_path}")

            except Exception as e:
                logger.error(f"Failed to evaluate modality {modality}: {e}")
                import traceback
                traceback.print_exc()
                all_modality_scores[modality] = {"error": str(e)}
            finally:
                # Restore original output path
                self.eval_config.encode_output_path = original_output_path

        # Save overall summary
        overall_summary_path = os.path.join(self.eval_config.encode_output_path, "all_modalities_summary.json")
        with open(overall_summary_path, 'w') as f:
            json.dump(all_modality_scores, f, indent=2)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"All modalities evaluation complete!")
        logger.info(f"Overall summary saved to {overall_summary_path}")
        logger.info("=" * 60)

        # Print summary statistics
        self._print_summary_statistics(all_modality_scores)

        return all_modality_scores

    def _print_summary_statistics(self, all_modality_scores: Dict):
        """Print summary statistics for all modalities"""
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)

        for modality, scores in all_modality_scores.items():
            if isinstance(scores, dict) and "error" not in scores:
                # Calculate average precision@1 for this modality
                p1_scores = []
                for dataset_name, dataset_scores in scores.items():
                    if isinstance(dataset_scores, dict) and "precision@1" in dataset_scores:
                        p1_scores.append(dataset_scores["precision@1"])

                if p1_scores:
                    avg_p1 = sum(p1_scores) / len(p1_scores)
                    logger.info(f"{modality.upper():10s}: {len(scores):3d} datasets, avg precision@1 = {avg_p1:.4f}")
                else:
                    logger.info(f"{modality.upper():10s}: {len(scores):3d} datasets")
            else:
                logger.info(f"{modality.upper():10s}: ERROR - {scores.get('error', 'Unknown error')}")


# ============================================================================
# Command Line Interface
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="VLM2Vec ByteDance Ark API-based Evaluation")

    # API settings
    parser.add_argument("--api_base_url", type=str,
                        default="https://ark-cn-beijing.bytedance.net/api/v3/embeddings/multimodal",
                        help="ByteDance Ark API URL")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Ark API key (or set ARK_API_KEY env var)")
    parser.add_argument("--model", type=str, default="ep-20250917160742-f9hzv",
                        help="Ark model endpoint ID")
    parser.add_argument("--timeout", type=int, default=120,
                        help="API request timeout in seconds")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of retries for failed requests")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for API requests (Ark API processes one at a time)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers for API requests")

    # Evaluation settings
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Path to dataset config YAML file (not required if --eval_all_modalities is set)")
    parser.add_argument("--eval_all_modalities", action="store_true",
                        help="Evaluate all modalities (image, video, visdoc) at once")
    parser.add_argument("--modalities", type=str, nargs="+", default=None,
                        choices=["image", "video", "visdoc"],
                        help="Specific modalities to evaluate (default: all)")
    parser.add_argument("--data_basedir", type=str, default=None,
                        help="Base directory for datasets")
    parser.add_argument("--encode_output_path", type=str, default="./output/api_eval",
                        help="Output directory for embeddings and scores")
    parser.add_argument("--image_resolution", type=int, nargs=2, default=[672, 672],
                        help="Image resolution (width height)")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="Re-evaluate even if scores exist")

    # Single dataset evaluation
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Evaluate only this dataset (optional)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get("ARK_API_KEY", "")
    if not api_key:
        logger.error("API key not provided. Set --api_key or ARK_API_KEY environment variable.")
        sys.exit(1)

    # Validate arguments
    if not args.eval_all_modalities and not args.dataset_config and not args.modalities:
        logger.error("Either --eval_all_modalities, --modalities, or --dataset_config must be provided.")
        sys.exit(1)

    # Create configs
    api_config = APIConfig(
        api_base_url=args.api_base_url,
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    eval_config = EvalConfig(
        dataset_config=args.dataset_config,
        data_basedir=args.data_basedir,
        encode_output_path=args.encode_output_path,
        image_resolution=tuple(args.image_resolution),
        skip_existing=not args.no_skip_existing,
        eval_all_modalities=args.eval_all_modalities,
    )

    # Create evaluator
    evaluator = APIEvaluator(api_config, eval_config)

    # Run evaluation based on mode
    if args.eval_all_modalities or args.modalities:
        # Evaluate all or specific modalities
        if args.modalities:
            # Filter to only specified modalities
            global ALL_MODALITY_CONFIGS
            original_configs = ALL_MODALITY_CONFIGS.copy()
            ALL_MODALITY_CONFIGS = {k: v for k, v in ALL_MODALITY_CONFIGS.items() if k in args.modalities}
            logger.info(f"Evaluating selected modalities: {list(ALL_MODALITY_CONFIGS.keys())}")

        evaluator.evaluate_all_modalities()

        if args.modalities:
            ALL_MODALITY_CONFIGS = original_configs

    elif args.dataset_name:
        # Single dataset
        with open(args.dataset_config, 'r') as f:
            dataset_configs = yaml.safe_load(f)

        if args.dataset_name not in dataset_configs:
            logger.error(f"Dataset {args.dataset_name} not found in config")
            sys.exit(1)

        evaluator.evaluate_dataset(args.dataset_name, dataset_configs[args.dataset_name])
    else:
        # All datasets in single config file
        all_scores = evaluator.evaluate_all()

        # Save summary
        summary_path = os.path.join(eval_config.encode_output_path, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_scores, f, indent=2)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

