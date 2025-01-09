# src/pipeline/model_processor.py
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from src.utils.model_loader import ModelLoader
from src.preprocessing.mesh_normalization import MeshNormalizer
from src.descriptors.fourier import FourierDescriptor
from src.descriptors.zernike import ZernikeDescriptor

class ModelProcessor:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.loader = ModelLoader(data_dir)
        self.normalizer = MeshNormalizer(target_vertices=1000)
        self.fourier_desc = FourierDescriptor(resolution=32)
        self.zernike_desc = ZernikeDescriptor(max_order=10)
        self.logger = logging.getLogger(__name__)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def process_single_model(self, model_path: Path) -> Optional[Dict]:
        try:
            mesh = self.loader.load_model_safe(model_path)
            if mesh is None:
                self.logger.error(f"Failed to load mesh: {model_path}")
                return None

            normalized_result = self.normalizer.normalize(mesh)
            if normalized_result is None or len(normalized_result) != 2:
                self.logger.error(f"Failed to normalize mesh: {model_path}")
                return None

            normalized_mesh, voxel_data = normalized_result
            if voxel_data is None or 'voxel_grid' not in voxel_data:
                self.logger.error(f"Missing voxel data: {model_path}")
                return None

            voxel_grid = voxel_data['voxel_grid']
            if voxel_grid is None:
                self.logger.error(f"Invalid voxel grid: {model_path}")
                return None

            fourier_features = self.fourier_desc.compute(voxel_grid)
            zernike_features = self.zernike_desc.compute(voxel_grid)

            if fourier_features is None or zernike_features is None:
                self.logger.error(f"Failed to compute descriptors: {model_path}")
                return None

            return {
                'path': str(model_path),
                'fourier': fourier_features,
                'zernike': zernike_features
            }

        except Exception as e:
            self.logger.error(f"Error processing {model_path}: {str(e)}")
            return None

    def process_database(self, batch_size: int = 4) -> List[Dict]:
        model_files = []
        for ext in ['.obj', '.ply', '.stl', '.off']:
            found = list(self.data_dir.glob(f"**/*{ext}"))
            model_files.extend(found)
            self.logger.info(f"Found {len(found)} {ext} files")
        self.logger.info(f"Total models to process: {len(model_files)}")

        results = []
        for model_path in tqdm(model_files, desc="Processing Models"):
            self.logger.info(f"Processing: {model_path}")
            result = self.process_single_model(model_path)
            if result is not None:
                self.logger.info(f"Successfully processed: {model_path}")
                results.append(result)
            else:
                self.logger.warning(f"Failed to process: {model_path}")

        self.logger.info(f"Successfully processed {len(results)} models")
        return results

if __name__ == "__main__":
    from src.search_engine.model_search_engine import ModelSearchEngine

    # Initialize processor and process models
    processor = ModelProcessor(Path("data/raw_models/3D Models/All Models"))
    results = processor.process_database()

    # Create and save search index
    search_engine = ModelSearchEngine(Path("data/search_index.pkl"))
    search_engine.build_index(results)
    search_engine.save_index()