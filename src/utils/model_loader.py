import os
import numpy as np
from pathlib import Path
import trimesh
import logging
import sys
import traceback
import gc

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.supported_extensions = ['.obj', '.ply', '.stl', '.off']
        self.max_vertices = 1000000

    def get_all_model_paths(self):
        try:
            model_paths = []
            for ext in self.supported_extensions:
                found_models = list(self.dataset_path.glob(f"**/*{ext}"))
                logger.info(f"Found {len(found_models)} models with extension {ext}")
                model_paths.extend(found_models)
            return model_paths
        except Exception as e:
            logger.error(f"Error scanning directory: {str(e)}")
            return []

    def validate_mesh(self, mesh):
        """
        Validate mesh using trimesh's built-in checks
        """
        try:
            # Check if mesh has valid data
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                logger.error("Mesh has no vertices or faces")
                return False

            # Check for degenerate faces
            if not mesh.is_watertight:
                logger.warning("Mesh is not watertight")

            # Check if vertices are finite
            if not np.all(np.isfinite(mesh.vertices)):
                logger.error("Mesh contains invalid vertex coordinates")
                return False

            # Check if faces are valid (no repeated indices)
            if not np.all(mesh.faces >= 0):
                logger.error("Mesh contains invalid face indices")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating mesh: {str(e)}")
            return False

    def load_model_safe(self, model_path):
        """
        Load model using trimesh with proper validation
        """
        try:
            logger.debug(f"Attempting to load model: {model_path}")
            gc.collect()

            # Load with trimesh
            mesh = trimesh.load(
                str(model_path),
                force='mesh',
                process=False,
                validate=True
            )

            logger.debug(f"Mesh loaded with {len(mesh.vertices)} vertices")

            # Validate the mesh
            if self.validate_mesh(mesh):
                return mesh
            return None

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def analyze_mesh(self, mesh):
        """
        Analyze mesh properties using trimesh
        """
        try:
            if mesh is None:
                logger.error("No mesh to analyze")
                return None

            # Collect basic statistics
            stats = {
                "vertices_count": len(mesh.vertices),
                "faces_count": len(mesh.faces),
                "bounds": mesh.bounds.tolist(),
                "center_mass": mesh.center_mass.tolist(),
                "volume": float(mesh.volume) if mesh.is_watertight else None,
                "surface_area": float(mesh.area),
                "is_watertight": mesh.is_watertight,
                "euler_number": int(mesh.euler_number),
                "geometric_center": mesh.centroid.tolist()
            }

            # Additional shape analysis
            try:
                stats["principal_inertia_components"] = mesh.principal_inertia_components.tolist()
                stats["principal_axes"] = mesh.principal_inertia_vectors.tolist()
            except:
                logger.warning("Could not compute inertia properties")

            # Log statistics
            logger.info("Mesh Analysis:")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")

            return stats

        except Exception as e:
            logger.error(f"Error analyzing mesh: {str(e)}")
            return None

    def export_mesh(self, mesh, output_path, file_format='.ply'):
        """
        Export mesh to a file
        """
        try:
            if mesh is None:
                logger.error("No mesh to export")
                return False

            output_path = Path(output_path)
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True)

            mesh.export(str(output_path.with_suffix(file_format)))
            logger.info(f"Mesh exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting mesh: {str(e)}")
            return False

def test_load_and_analyze(model_path):
    """
    Test function to load and analyze a single model
    """
    try:
        logger.info(f"Testing model: {model_path}")
        logger.info(f"File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

        loader = ModelLoader(model_path.parent)

        # Load the model
        mesh = loader.load_model_safe(model_path)

        if mesh is not None:
            logger.info("Model loaded successfully")

            # Analyze the mesh
            stats = loader.analyze_mesh(mesh)

            # Export to a different format (optional)
            output_path = Path("test_output.ply")
            loader.export_mesh(mesh, output_path)

            return stats
        else:
            logger.error("Failed to load model")
            return None

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Set up test path
    test_path = Path("/home/hafdaoui/Desktop/Multimedia mining and indexing/3d_model_search/data/raw_models/3D Models")

    # Get all models
    loader = ModelLoader(test_path)
    models = loader.get_all_model_paths()
    print(f"Found {len(models)} models")

    if models:
        # Test first model
        stats = test_load_and_analyze(models[0])
        if stats:
            print("\nMesh Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")