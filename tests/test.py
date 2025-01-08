import pytest
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from src.utils.model_loader import ModelLoader
from src.preprocessing.mesh_normalization import MeshNormalizer
from src.descriptors.fourier import FourierDescriptor
from src.descriptors.zernike import ZernikeDescriptor
import gc
import logging
import numpy as np
import traceback
import sys

# Configure logging to show more details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

@pytest.fixture
def model_paths():
    base_path = Path("/home/hafdaoui/Desktop/Multimedia mining and indexing/3d_model_search/data/raw_models/3D Models")
    paths = []

    # Search for models recursively
    for ext in ['.obj', '.ply', '.stl', '.off']:
        found_paths = list(base_path.glob(f"**/*{ext}"))
        logger.info(f"Found {len(found_paths)} models with extension {ext}")
        paths.extend(found_paths)

    if not paths:
        logger.error(f"No models found in {base_path}")
        raise ValueError(f"No models found in {base_path}")

    # Log the actual paths being tested
    for path in paths[:8]:
        logger.info(f"Will process: {path}")

    return paths[:8]

def process_single_model(model_path):
    logger.info(f"\n{'='*50}\nStarting to process model: {model_path}\n{'='*50}")

    try:
        # Step 1: Initialize processors
        logger.debug("Initializing processors...")
        loader = ModelLoader(model_path.parent)
        normalizer = MeshNormalizer(target_vertices=1000)
        fourier_desc = FourierDescriptor(resolution=32)
        zernike_desc = ZernikeDescriptor(max_order=10)
        logger.debug("Processors initialized successfully")

        # Step 2: Load mesh
        logger.debug("Loading mesh...")
        mesh = loader.load_model_safe(model_path)
        if mesh is None:
            logger.error(f"Failed to load mesh: {model_path}")
            return None
        logger.info(f"Successfully loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

        # Step 3: Normalize mesh
        logger.debug("Normalizing mesh...")
        normalized_result = normalizer.normalize(mesh)
        if normalized_result is None or len(normalized_result) != 2:
            logger.error("Normalization returned invalid result")
            return None

        normalized_mesh, voxel_data = normalized_result
        if normalized_mesh is None or voxel_data is None:
            logger.error("Normalization failed - null results")
            return None
        logger.info("Mesh normalized successfully")

        # Step 4: Process voxel grid
        logger.debug("Processing voxel grid...")
        voxel_grid = voxel_data['voxel_grid']

        # Validate voxel grid
        if voxel_grid is None or voxel_grid.size == 0:
            logger.error("Invalid voxel grid - empty or None")
            return None

        if not np.any(voxel_grid):
            logger.error("Voxel grid contains all zeros")
            return None

        logger.info(f"Initial voxel grid shape: {voxel_grid.shape}")
        logger.info(f"Voxel grid range: [{np.min(voxel_grid)}, {np.max(voxel_grid)}]")

        # Resize voxel grid if necessary
        if voxel_grid.shape != (32, 32, 32):
            logger.debug("Resizing voxel grid...")
            from scipy.ndimage import zoom
            scale = np.array([32, 32, 32]) / np.array(voxel_grid.shape)
            voxel_grid = zoom(voxel_grid, scale, order=1)
            logger.info(f"Resized voxel grid to shape: {voxel_grid.shape}")

        # Step 5: Compute descriptors
        logger.debug("Computing Fourier descriptor...")
        try:
            fourier_features = fourier_desc.compute(voxel_grid)
            logger.info(f"Fourier features shape: {fourier_features.shape}")
            logger.info(f"Fourier descriptor values for {model_path}:\n{fourier_features}")
        except Exception as e:
            logger.error(f"Fourier descriptor computation failed: {str(e)}")
            return None

        logger.debug("Computing Zernike descriptor...")
        try:
            zernike_features = zernike_desc.compute(voxel_grid)
            logger.info(f"Zernike features shape: {zernike_features.shape}")
            logger.info(f"Zernike descriptor values for {model_path}:\n{zernike_features}")
        except Exception as e:
            logger.error(f"Zernike descriptor computation failed: {str(e)}")
            return None

        # Step 6: Prepare result
        result = {
            'path': str(model_path),
            'fourier': fourier_features,
            'zernike': zernike_features
        }

        logger.info(f"Successfully processed model: {model_path}")
        return result

    except Exception as e:
        logger.error(f"Error processing {model_path}:")
        logger.error(traceback.format_exc())
        return None
    finally:
        gc.collect()

def process_batch(model_paths, batch_size=4):
    logger.info(f"Starting batch processing of {len(model_paths)} models")

    # Process models sequentially first for debugging
    results = []
    for path in tqdm(model_paths):
        result = process_single_model(path)
        if result is not None:
            results.append(result)
            logger.info(f"Successfully processed: {path}")
        else:
            logger.error(f"Failed to process: {path}")

    logger.info(f"Successfully processed {len(results)} out of {len(model_paths)} models")
    return results

def test_pipeline(model_paths):
    logger.info("\nStarting pipeline test")

    # Verify input paths
    assert len(model_paths) > 0, "No model paths provided"
    logger.info(f"Testing with {len(model_paths)} models")

    # Process models
    results = process_batch(model_paths)

    # Basic validation
    assert len(results) > 0, "No models were successfully processed"

    # Detailed validation
    for result in results:
        assert isinstance(result, dict), f"Invalid result type: {type(result)}"
        assert 'path' in result, f"Missing path in result: {result}"
        assert 'fourier' in result, f"Missing Fourier descriptor in result: {result}"
        assert 'zernike' in result, f"Missing Zernike descriptor in result: {result}"

        assert isinstance(result['fourier'], np.ndarray), f"Invalid Fourier type: {type(result['fourier'])}"
        assert isinstance(result['zernike'], np.ndarray), f"Invalid Zernike type: {type(result['zernike'])}"

        logger.info(f"Validated result for: {result['path']}")

    logger.info("Pipeline test completed successfully")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])