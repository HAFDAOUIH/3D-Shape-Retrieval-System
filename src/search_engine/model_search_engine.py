# src/search_engine/model_search_engine.py

import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import logging

class ModelSearchEngine:
    def __init__(self, index_path: Path = None):
        self.index: Dict[str, Dict] = {}
        self.index_path = index_path
        self.logger = logging.getLogger(__name__)

    def add_model(self, model_path: str, fourier_desc: np.ndarray, zernike_desc: np.ndarray) -> None:
        """Add a model and its descriptors to the index."""
        self.index[model_path] = {
            'fourier': fourier_desc,
            'zernike': zernike_desc
        }

    def build_index(self, processed_results: List[Dict]) -> None:
        """Build search index from processed model results."""
        for result in processed_results:
            self.add_model(
                result['path'],
                result['fourier'],
                result['zernike']
            )
        self.logger.info(f"Built index with {len(self.index)} models")

    def save_index(self) -> None:
        """Save index to disk with proper directory creation."""
        try:
            if self.index_path:
                self.logger.debug(f"Creating directories for {self.index_path.parent}")
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Saving index to {self.index_path}")
                with open(self.index_path, 'wb') as f:
                    pickle.dump(self.index, f)
                self.logger.info(f"Index saved successfully to {self.index_path}")
        except Exception as e:
            self.logger.error(f"Failed to save index: {e}")


    def load_index(self) -> None:
        """Load index with proper error handling."""
        if not self.index_path:
            raise ValueError("No index path specified")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")
        try:
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            self.logger.info(f"Index loaded from {self.index_path} with {len(self.index)} entries")
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            raise


    def compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Compute cosine similarity between descriptors."""
        return np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))

    def search(self, query_fourier: np.ndarray, query_zernike: np.ndarray,
               top_k: int = 10, weights: Tuple[float, float] = (0.5, 0.5)) -> List[Tuple[str, float]]:
        """
        Search for similar models using weighted combination of descriptors.
        
        Args:
            query_fourier: Fourier descriptor of query model
            query_zernike: Zernike descriptor of query model
            top_k: Number of results to return
            weights: Weights for Fourier and Zernike descriptors
            
        Returns:
            List of (model_path, similarity_score) tuples
        """
        w_fourier, w_zernike = weights
        results = []

        for model_path, descriptors in self.index.items():
            fourier_sim = self.compute_similarity(query_fourier, descriptors['fourier'])
            zernike_sim = self.compute_similarity(query_zernike, descriptors['zernike'])

            # Weighted combination
            combined_sim = w_fourier * fourier_sim + w_zernike * zernike_sim
            results.append((model_path, combined_sim))

        # Sort by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]