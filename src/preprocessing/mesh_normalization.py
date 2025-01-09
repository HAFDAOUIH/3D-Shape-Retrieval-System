import numpy as np
import trimesh
import logging
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_fill_holes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json
import datetime


class VoxelProcessor:
    def __init__(self, resolution=64):
        self.resolution = resolution

    def create_voxel_grid(self, mesh, fill_interior=True):
        """
        Convert mesh to voxel grid with consistent dimensions and improved error handling.
        """
        try:
            if mesh is None:
                return None

            # Scale mesh to fit unit cube
            extents = mesh.extents
            scale = 1.0 / max(extents)
            mesh = mesh.copy()
            mesh.apply_scale(scale)

            # Create voxel grid
            voxels = mesh.voxelized(pitch=1.0/self.resolution)
            if voxels is None:
                return None

            # Fill the volume
            if fill_interior:
                try:
                    voxels = voxels.fill()
                except Exception as e:
                    logging.warning(f"Fill operation failed: {e}")
                    # Continue with unfilled voxels

            matrix = voxels.matrix
            if matrix is None:
                return None

            # Ensure consistent dimensions through padding/cropping
            target_shape = (self.resolution, self.resolution, self.resolution)
            result = np.zeros(target_shape, dtype=bool)

            # Calculate dimensions to copy
            dims = [min(s, t) for s, t in zip(matrix.shape, target_shape)]

            # Center the voxel data
            starts = [(t - d) // 2 for t, d in zip(target_shape, dims)]

            # Copy data to center of target array
            slices_target = tuple(slice(s, s + d) for s, d in zip(starts, dims))
            slices_source = tuple(slice(0, d) for d in dims)

            result[slices_target] = matrix[slices_source]

            return result

        except Exception as e:
            logging.error(f"Error in voxelization: {e}")
            return None


    def get_surface_voxels(self, voxel_grid):
        """
        Extract surface voxels from the grid.
        """
        surface = np.zeros_like(voxel_grid)

        # Mark voxels that have at least one empty neighbor
        padded = np.pad(voxel_grid, 1, mode='constant')
        for x in range(1, padded.shape[0]-1):
            for y in range(1, padded.shape[1]-1):
                for z in range(1, padded.shape[2]-1):
                    if padded[x,y,z]:
                        neighbors = [
                            padded[x-1,y,z], padded[x+1,y,z],
                            padded[x,y-1,z], padded[x,y+1,z],
                            padded[x,y,z-1], padded[x,y,z+1]
                        ]
                        if not all(neighbors):
                            surface[x-1,y-1,z-1] = 1

        return surface

    def process_mesh(self, mesh):
        """
        Complete processing pipeline with error handling.
        """
        voxel_grid = self.create_voxel_grid(mesh)
        if voxel_grid is None:
            return None

        try:
            surface_voxels = self.get_surface_voxels(voxel_grid)

            return {
                'voxel_grid': voxel_grid,
                'surface_voxels': surface_voxels,
                'resolution': self.resolution,
                'is_watertight': mesh.is_watertight if mesh is not None else False
            }
        except Exception as e:
            logging.error(f"Error in process_mesh: {e}")
            return None


class MeshNormalizer:
    def __init__(self, target_vertices=1000, make_watertight=True):
        """
        Initialize the enhanced mesh normalizer.

        Args:
            target_vertices (int): Target number of vertices after simplification
            make_watertight (bool): Whether to attempt making the mesh watertight
        """
        self.target_scale = 1.0
        self.center = True
        self.align_to_principal = True
        self.target_vertices = target_vertices
        self.make_watertight = make_watertight


    def visualize_comparison(self, original_mesh, normalized_mesh, save_path=None):
        """
        Create a side-by-side visualization of original and normalized meshes.

        Args:
            original_mesh (trimesh.Trimesh): Original mesh
            normalized_mesh (trimesh.Trimesh): Normalized mesh
            save_path (str, optional): Path to save the visualization
        """
        try:
            # Create figure with two subplots
            fig = plt.figure(figsize=(15, 7))

            # Original mesh plot
            ax1 = fig.add_subplot(121, projection='3d')
            self._plot_mesh(original_mesh, ax1, "Original Mesh")

            # Normalized mesh plot
            ax2 = fig.add_subplot(122, projection='3d')
            self._plot_mesh(normalized_mesh, ax2, "Normalized Mesh")

            # Adjust layout
            plt.tight_layout()

            # Save or show
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")

    def _plot_mesh(self, mesh, ax, title):
        """
        Helper method to plot a single mesh.

        Args:
            mesh (trimesh.Trimesh): Mesh to plot
            ax (Axes3D): Matplotlib 3D axes
            title (str): Plot title
        """
        try:
            # Get vertices and faces
            vertices = mesh.vertices
            faces = mesh.faces

            # Plot the mesh
            ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                            triangles=faces, alpha=0.8)

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)

            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])

        except Exception as e:
            logger.error(f"Error in mesh plotting: {str(e)}")

    def print_mesh_stats(self, mesh, prefix=""):
        """
        Print detailed statistics about the mesh.

        Args:
            mesh (trimesh.Trimesh): Input mesh
            prefix (str): Prefix for the output (e.g., "Original" or "Normalized")
        """
        try:
            stats = {
                "Vertices": len(mesh.vertices),
                "Faces": len(mesh.faces),
                "Bounds": mesh.bounds,
                "Center of Mass": mesh.center_mass,
                "Surface Area": mesh.area,
                "Is Watertight": mesh.is_watertight,
                "Euler Number": mesh.euler_number
            }

            logger.info(f"\n{prefix} Mesh Statistics:")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")

        except Exception as e:
            logger.error(f"Error calculating mesh statistics: {str(e)}")

    def normalize_scale(self, mesh):
        """Normalize the mesh scale to a unit cube while preserving aspect ratio."""
        try:
            extents = mesh.extents
            max_extent = np.max(extents)
            scale_factor = self.target_scale / max_extent
            matrix = np.eye(4)
            matrix[:3, :3] *= scale_factor
            mesh = mesh.apply_transform(matrix)
            logger.info(f"Scaled mesh by factor {scale_factor:.4f}")
            return mesh
        except Exception as e:
            logger.error(f"Error in scale normalization: {str(e)}")
            return None

    def center_mesh(self, mesh):
        """Center the mesh at origin using center of mass."""
        try:
            center = mesh.centroid
            matrix = np.eye(4)
            matrix[:3, 3] = -center
            mesh = mesh.apply_transform(matrix)
            logger.info(f"Centered mesh at origin (offset: {center})")
            return mesh
        except Exception as e:
            logger.error(f"Error in mesh centering: {str(e)}")
            return None

    def align_principal_axes(self, mesh):
        """Align mesh to its principal axes."""
        try:
            inertia_transform = mesh.principal_inertia_transform
            mesh = mesh.apply_transform(inertia_transform)
            logger.info("Aligned mesh to principal axes")
            return mesh
        except Exception as e:
            logger.error(f"Error in principal axes alignment: {str(e)}")
            return None

    def fill_holes(self, mesh):
        """
        Attempt to fill holes in the mesh with improved error handling.
        """
        try:
            # First try using trimesh's built-in fill holes
            filled_mesh = trimesh.repair.fill_holes(mesh)
            if filled_mesh is not None and isinstance(filled_mesh, trimesh.Trimesh):
                return filled_mesh

            # If that fails, try alternative approach
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()

            # Find boundary edges
            edges = mesh.edges_unique
            boundary_edges = mesh.edges_unique[mesh.edges_unique_length == 1]

            if len(boundary_edges) > 0:
                # Simple triangulation of small holes
                for edge in boundary_edges:
                    if len(faces) < len(vertices) * 3:  # Basic safety check
                        # Create new triangle using nearest vertex
                        distances = np.linalg.norm(vertices - vertices[edge[0]], axis=1)
                        nearest = np.argmin(distances[~np.isin(np.arange(len(vertices)), edge)])
                        new_face = np.array([edge[0], edge[1], nearest])
                        faces = np.vstack((faces, new_face))

                new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                new_mesh.remove_duplicate_faces()
                new_mesh.remove_degenerate_faces()
                return new_mesh

            return mesh

        except Exception as e:
            logger.warning(f"Error in hole filling: {str(e)}")
            return mesh


    def fix_mesh(self, mesh):
        """
        Enhanced mesh fixing with better error handling.
        """
        try:
            if mesh is None or not isinstance(mesh, trimesh.Trimesh):
                logger.error("Invalid mesh input")
                return None

            # Create a copy to avoid modifying original
            mesh = mesh.copy()

            # Basic cleanup
            try:
                mesh.process(validate=True)
            except Exception as e:
                logger.warning(f"Basic cleanup failed: {str(e)}")

            # Merge close vertices
            try:
                mesh.merge_vertices(merge_tex=True, merge_norm=True)
            except Exception as e:
                logger.warning(f"Vertex merging failed: {str(e)}")

            # Update faces
            try:
                nondegenerate = mesh.nondegenerate_faces()
                if len(nondegenerate) > 0:
                    mesh.update_faces(nondegenerate)
                unique_faces = mesh.unique_faces()
                if len(unique_faces) > 0:
                    mesh.update_faces(unique_faces)
            except Exception as e:
                logger.warning(f"Face updating failed: {str(e)}")

            # Remove disconnected components
            try:
                components = mesh.split(only_watertight=False)
                if len(components) > 1:
                    areas = np.array([c.area for c in components])
                    mesh = components[np.argmax(areas)]
                    logger.info(f"Removed {len(components)-1} disconnected components")
            except Exception as e:
                logger.warning(f"Component removal failed: {str(e)}")

            # Make watertight if requested
            if self.make_watertight and not mesh.is_watertight:
                try:
                    logger.info("Attempting to make mesh watertight...")
                    mesh = self.fill_holes(mesh)
                except Exception as e:
                    logger.warning(f"Watertight conversion failed: {str(e)}")

            # Ensure consistent face winding
            try:
                mesh.fix_normals()
            except Exception as e:
                logger.warning(f"Normal fixing failed: {str(e)}")

            logger.info("Applied enhanced mesh fixing operations")
            return mesh

        except Exception as e:
            logger.error(f"Error in mesh fixing: {str(e)}")
            return None

    def simplify_mesh(self, mesh):
        """
        Simplify mesh to target number of vertices while preserving shape.
        """
        try:
            current_vertices = len(mesh.vertices)
            if current_vertices > self.target_vertices:
                ratio = self.target_vertices / current_vertices
                mesh = mesh.simplify_quadratic_decimation(int(len(mesh.faces) * ratio))
                logger.info(f"Simplified mesh from {current_vertices} to {len(mesh.vertices)} vertices")
            return mesh
        except Exception as e:
            logger.error(f"Error in mesh simplification: {str(e)}")
            return None

    def sample_surface(self, mesh, n_points=10000):
        """
        Sample points uniformly from the mesh surface.
        """
        try:
            points, _ = trimesh.sample.sample_surface(mesh, n_points)
            return points
        except Exception as e:
            logger.error(f"Error in surface sampling: {str(e)}")
            return None

    def normalize(self, mesh, fix_mesh=True):
        """
        Enhanced normalization pipeline with better error handling.
        Includes voxelization processing.
        """
        try:
            if mesh is None:
                logger.error("Input mesh is None")
                return None, None, None

            logger.info("Starting enhanced mesh normalization")

            # Store original mesh for comparison
            original_mesh = mesh.copy()

            # Print original stats
            self.print_mesh_stats(original_mesh, "Original")

            # Fix mesh if requested
            if fix_mesh:
                mesh = self.fix_mesh(mesh)
                if mesh is None:
                    logger.error("Mesh fixing failed")
                    return None, None, None

            # Apply other normalizations with error checking
            if self.center:
                mesh = self.center_mesh(mesh)
                if mesh is None:
                    return None, None, None

            if self.align_to_principal:
                mesh = self.align_principal_axes(mesh)
                if mesh is None:
                    return None, None, None

            mesh = self.normalize_scale(mesh)
            if mesh is None:
                return None, None, None

            # Sample surface points
            try:
                surface_points = self.sample_surface(mesh)
                if surface_points is not None:
                    logger.info(f"Sampled {len(surface_points)} surface points")
            except Exception as e:
                logger.warning(f"Surface sampling failed: {str(e)}")
                surface_points = None

            # Add voxelization step
            try:
                voxel_processor = VoxelProcessor(resolution=64)
                voxel_data = voxel_processor.process_mesh(mesh)
                logger.info(f"Created voxel grid with resolution {voxel_data['resolution']}^3")
                logger.info(f"Voxel grid shape: {voxel_data['voxel_grid'].shape}")
            except Exception as e:
                logger.warning(f"Voxelization failed: {str(e)}")
                voxel_data = None

            # Print normalized stats
            self.print_mesh_stats(mesh, "Normalized")

            return mesh, voxel_data

        except Exception as e:
            logger.error(f"Error in normalization pipeline: {str(e)}")
            return None, None, None

def process_all_models(input_dir, output_dir):
    """
    Process all models in the input directory with improved error handling and reporting.
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add statistics tracking
        stats = {
            'total': 0,
            'success': 0,
            'watertight_fixed': 0,
            'failed': 0,
            'errors': {}
        }

        from src.utils.model_loader import ModelLoader
        loader = ModelLoader(input_dir)
        models = loader.get_all_model_paths()

        if not models:
            logger.error("No models found for processing")
            return

        normalizer = MeshNormalizer(target_vertices=2000, make_watertight=True)
        failed_models = []

        # Add error type tracking
        error_types = {}

        for model_path in tqdm(models, desc="Processing models"):
            stats['total'] += 1
            try:
                # Load model with additional error catching
                try:
                    mesh = loader.load_model_safe(model_path)
                except Exception as e:
                    error_msg = f"Model loading failed: {str(e)}"
                    error_types[str(type(e).__name__)] = error_types.get(str(type(e).__name__), 0) + 1
                    failed_models.append((str(model_path), error_msg))
                    continue

                if mesh is None:
                    failed_models.append((str(model_path), "Mesh loading returned None"))
                    continue

                # Track initial watertight status
                was_watertight = mesh.is_watertight

                # Normalize mesh with additional metadata
                normalized_mesh, surface_points, voxel_data = normalizer.normalize(mesh)

                if normalized_mesh is not None:
                    # Track watertight fixing success
                    if not was_watertight and normalized_mesh.is_watertight:
                        stats['watertight_fixed'] += 1

                    # Create output paths
                    base_filename = Path(model_path).stem
                    mesh_output = output_dir / f"normalized_{base_filename}.ply"
                    points_output = output_dir / f"points_{base_filename}.npy"
                    voxel_output = output_dir / f"voxels_{base_filename}.npz"
                    metadata_output = output_dir / f"metadata_{base_filename}.json"

                    # Export all data
                    normalized_mesh.export(str(mesh_output))

                    if surface_points is not None:
                        np.save(str(points_output), surface_points)

                    if voxel_data is not None:
                        np.savez_compressed(str(voxel_output),
                                            voxel_grid=voxel_data['voxel_grid'],
                                            surface_voxels=voxel_data['surface_voxels'])

                    # Save processing metadata
                    metadata = {
                        'original_path': str(model_path),
                        'vertices_count': len(normalized_mesh.vertices),
                        'faces_count': len(normalized_mesh.faces),
                        'is_watertight': normalized_mesh.is_watertight,
                        'surface_points': surface_points is not None,
                        'voxel_grid': voxel_data is not None,
                        'processing_date': str(datetime.now())
                    }

                    with open(metadata_output, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    stats['success'] += 1
                else:
                    failed_models.append((str(model_path), "Normalization failed"))
                    stats['failed'] += 1

            except Exception as e:
                error_msg = f"Processing failed: {str(e)}"
                error_type = str(type(e).__name__)
                error_types[error_type] = error_types.get(error_type, 0) + 1
                failed_models.append((str(model_path), error_msg))
                stats['failed'] += 1

        # Generate detailed report
        report = {
            'processing_stats': stats,
            'error_types': error_types,
            'failed_models': failed_models
        }

        with open(output_dir / 'processing_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Log summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total models processed: {stats['total']}")
        logger.info(f"Successfully processed: {stats['success']}")
        logger.info(f"Watertight fixes: {stats['watertight_fixed']}")
        logger.info(f"Failed models: {stats['failed']}")

        if error_types:
            logger.info("\nError type distribution:")
            for error_type, count in error_types.items():
                logger.info(f"  {error_type}: {count}")

    except Exception as e:
        logger.error(f"Fatal error in batch processing: {str(e)}")
        raise

if __name__ == "__main__":
    input_dir = Path("/home/hafdaoui/Desktop/Multimedia mining and indexing/3d_model_search/data/raw_models/3D Models")
    output_dir = Path("/home/hafdaoui/Desktop/Multimedia mining and indexing/3d_model_search/data/normalized_models")
    process_all_models(input_dir, output_dir)