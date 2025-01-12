import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
import time
from scipy import stats
import seaborn as sns
class DescriptorAnalyzer:
    def __init__(self, processor, search_engine):
        self.processor = processor
        self.search_engine = search_engine
        self.results = []
        self.comparative_results = {}

    def analyze_with_different_resolutions(self, test_models: List[Path],
                                           resolutions: List[int] = [500, 1000, 2000]) -> Dict:
        """
        Analyze models with different mesh resolutions.
        """
        results = {
            'fourier': {},
            'zernike': {},
            'processing_times': {},
            'memory_usage': {}
        }

        for resolution in resolutions:
            print(f"\nAnalyzing with resolution: {resolution}")
            self.processor.normalizer.target_vertices = resolution

            resolution_results = []
            start_time = time.time()

            for model_path in test_models:
                metrics = self.processor.process_single_model(model_path)
                if metrics:
                    resolution_results.append(metrics)

            processing_time = time.time() - start_time

            # Store results for this resolution
            if resolution_results:  # Only store if we have results
                results['fourier'][resolution] = [r['fourier'] for r in resolution_results]
                results['zernike'][resolution] = [r['zernike'] for r in resolution_results]
                results['processing_times'][resolution] = processing_time

        return results

    def compute_similarity_metrics(self, query_results: List[Tuple[str, float]],
                                   ground_truth: List[str]) -> Dict:
        """
        Compute precision, recall, and other similarity metrics.
        """
        if not query_results or not ground_truth:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'num_retrieved': 0,
                'num_relevant': 0
            }

        retrieved_models = [result[0] for result in query_results]
        relevant_retrieved = set(retrieved_models) & set(ground_truth)

        precision = len(relevant_retrieved) / len(retrieved_models) if retrieved_models else 0
        recall = len(relevant_retrieved) / len(ground_truth) if ground_truth else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'num_retrieved': len(retrieved_models),
            'num_relevant': len(relevant_retrieved)
        }

    def analyze_descriptor_distributions(self, results: Dict, output_dir: Path) -> Dict:
        """
        Analyze the statistical distributions of descriptors.
        """
        distribution_metrics = {
            'fourier': {},
            'zernike': {}
        }

        # Create directory for distribution plots
        dist_dir = output_dir / 'distributions'
        dist_dir.mkdir(parents=True, exist_ok=True)

        for resolution in results['fourier'].keys():
            # Fourier descriptor analysis
            fourier_data = np.array(results['fourier'][resolution])
            zernike_data = np.array(results['zernike'][resolution])

            # Calculate distribution metrics
            distribution_metrics['fourier'][resolution] = {
                'mean': np.mean(fourier_data),
                'std': np.std(fourier_data),
                'skewness': stats.skew(fourier_data.flatten()),
                'kurtosis': stats.kurtosis(fourier_data.flatten()),
                'percentiles': np.percentile(fourier_data, [25, 50, 75])
            }

            distribution_metrics['zernike'][resolution] = {
                'mean': np.mean(zernike_data),
                'std': np.std(zernike_data),
                'skewness': stats.skew(zernike_data.flatten()),
                'kurtosis': stats.kurtosis(zernike_data.flatten()),
                'percentiles': np.percentile(zernike_data, [25, 50, 75])
            }

            # Generate separate distribution plots for each descriptor
            plt.figure(figsize=(15, 5))

            # Fourier distribution
            plt.subplot(121)
            sns.histplot(fourier_data.flatten(), kde=True)
            plt.title(f'Fourier Descriptor Distribution\nResolution: {resolution}')
            plt.xlabel('Descriptor Value')
            plt.ylabel('Frequency')

            # Zernike distribution
            plt.subplot(122)
            sns.histplot(zernike_data.flatten(), kde=True)
            plt.title(f'Zernike Moment Distribution\nResolution: {resolution}')
            plt.xlabel('Moment Value')
            plt.ylabel('Frequency')

            plt.tight_layout()
            plt.savefig(dist_dir / f'descriptor_distribution_{resolution}.png')
            plt.close()

        return distribution_metrics

    def analyze_descriptor_correlations(self, results: Dict, output_dir: Path) -> Dict:
        """
        Analyze correlations between different descriptors, handling different dimensions.
        """
        correlation_metrics = {}

        for resolution in results['fourier'].keys():
            fourier_data = np.array(results['fourier'][resolution])
            zernike_data = np.array(results['zernike'][resolution])

            # Get the minimum length between both descriptors
            min_length = min(fourier_data.shape[1], zernike_data.shape[1])

            # Trim both arrays to the same length
            fourier_trimmed = fourier_data[:, :min_length]
            zernike_trimmed = zernike_data[:, :min_length]

            # Calculate mean vectors
            fourier_mean = fourier_trimmed.mean(axis=0)
            zernike_mean = zernike_trimmed.mean(axis=0)

            # Calculate correlation between mean vectors
            correlation = np.corrcoef(fourier_mean, zernike_mean)[0, 1]

            correlation_metrics[resolution] = {
                'fourier_zernike_correlation': correlation,
                'fourier_dim': fourier_data.shape[1],
                'zernike_dim': zernike_data.shape[1],
                'analyzed_dim': min_length
            }

            # Generate correlation visualization for the trimmed data
            plt.figure(figsize=(10, 8))

            # Create correlation matrix for visualization
            corr_matrix = np.zeros((2, 2))
            corr_matrix[0, 0] = 1  # Fourier self-correlation
            corr_matrix[1, 1] = 1  # Zernike self-correlation
            corr_matrix[0, 1] = correlation  # Cross-correlation
            corr_matrix[1, 0] = correlation  # Symmetric cross-correlation

            # Plot correlation heatmap
            sns.heatmap(corr_matrix,
                        cmap='coolwarm',
                        center=0,
                        xticklabels=['Fourier', 'Zernike'],
                        yticklabels=['Fourier', 'Zernike'],
                        annot=True,
                        fmt='.2f')
            plt.title(f'Descriptor Correlation Heatmap\nResolution: {resolution}')
            plt.savefig(output_dir / f'correlation_heatmap_{resolution}.png')
            plt.close()

        return correlation_metrics

    def generate_comparative_visualizations(self, results: Dict, output_dir: Path) -> None:
        """
        Generate comprehensive comparative visualizations.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if not results['processing_times']:
            logging.warning("No results to visualize")
            return

        # 1. Processing Time Comparison
        plt.figure(figsize=(10, 6))
        resolutions = list(results['processing_times'].keys())
        times = list(results['processing_times'].values())
        plt.bar(resolutions, times)
        plt.title('Processing Time vs Mesh Resolution')
        plt.xlabel('Number of Vertices')
        plt.ylabel('Processing Time (seconds)')
        plt.savefig(output_dir / 'processing_times_comparison.png')
        plt.close()

        # 2. Descriptor Distribution Comparison
        plt.figure(figsize=(15, 6))

        # Fourier descriptors
        plt.subplot(1, 2, 1)
        for resolution in resolutions:
            if results['fourier'][resolution]:  # Check if we have data
                fourier_data = np.mean(results['fourier'][resolution], axis=0)
                plt.plot(fourier_data[:10], label=f'{resolution} vertices')
        plt.title('Average Fourier Descriptor Values')
        plt.xlabel('Coefficient Index')
        plt.ylabel('Magnitude')
        plt.legend()

        # Zernike moments
        plt.subplot(1, 2, 2)
        for resolution in resolutions:
            if results['zernike'][resolution]:  # Check if we have data
                zernike_data = np.mean(results['zernike'][resolution], axis=0)
                plt.plot(zernike_data[:10], label=f'{resolution} vertices')
        plt.title('Average Zernike Moment Values')
        plt.xlabel('Moment Index')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'descriptor_comparison.png')
        plt.close()


    def generate_analysis_metrics(self, results: Dict, output_dir: Path) -> Dict:
        """
        Generate comprehensive analysis metrics.
        """
        metrics = {
            'distributions': self.analyze_descriptor_distributions(results, output_dir),
            'correlations': self.analyze_descriptor_correlations(results, output_dir),
            'performance': self._analyze_performance_metrics(results)
        }

        # Save metrics to file
        metrics_file = output_dir / 'analysis_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write("Analysis Metrics Report\n")
            f.write("=====================\n\n")

            # Write distribution metrics
            f.write("Distribution Metrics:\n")
            f.write("-------------------\n")
            for desc_type in ['fourier', 'zernike']:
                f.write(f"\n{desc_type.capitalize()} Descriptors:\n")
                for resolution, stats in metrics['distributions'][desc_type].items():
                    f.write(f"\nResolution {resolution}:\n")
                    f.write(f"  Mean: {stats['mean']:.4f}\n")
                    f.write(f"  Std Dev: {stats['std']:.4f}\n")
                    f.write(f"  Skewness: {stats['skewness']:.4f}\n")
                    f.write(f"  Kurtosis: {stats['kurtosis']:.4f}\n")
                    f.write(f"  Percentiles (25, 50, 75): {stats['percentiles']}\n")

            # Write correlation metrics
            f.write("\nCorrelation Metrics:\n")
            f.write("-------------------\n")
            for resolution, corr in metrics['correlations'].items():
                f.write(f"\nResolution {resolution}:\n")
                f.write(f"  Fourier-Zernike Correlation: {corr['fourier_zernike_correlation']:.4f}\n")
                f.write(f"  Original Dimensions - Fourier: {corr['fourier_dim']}, Zernike: {corr['zernike_dim']}\n")
                f.write(f"  Analyzed Dimensions: {corr['analyzed_dim']}\n")

            # Write performance metrics
            f.write("\nPerformance Metrics:\n")
            f.write("-------------------\n")
            for resolution, perf in metrics['performance'].items():
                f.write(f"\nResolution {resolution}:\n")
                f.write(f"  Processing Time: {perf['processing_time']:.4f} seconds\n")
                f.write(f"  Memory Usage: {perf['memory_usage']:.2f} MB\n")

        return metrics

    def _format_processing_analysis(self, processing_times: Dict) -> str:
        """
        Format processing time analysis results.
        """
        if not processing_times:
            return "No processing time data available."

        analysis = ["Processing Time Analysis:"]
        for resolution, time in processing_times.items():
            analysis.append(f"- Resolution {resolution}: {time:.2f} seconds")

        # Add performance insights
        min_time = min(processing_times.values())
        max_time = max(processing_times.values())
        avg_time = sum(processing_times.values()) / len(processing_times)

        analysis.extend([
            f"\nPerformance Summary:",
            f"- Fastest processing time: {min_time:.2f} seconds",
            f"- Slowest processing time: {max_time:.2f} seconds",
            f"- Average processing time: {avg_time:.2f} seconds",
            f"- Processing time variation: {(max_time - min_time):.2f} seconds"
        ])

        return "\n".join(analysis)

    def generate_detailed_report(self, results: Dict, output_dir: Path) -> None:
        """
        Generate a comprehensive analysis report.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        report = f"""
3D Model Descriptor Comparative Analysis Report
============================================

1. Performance Analysis
----------------------
Resolution Impact:
{self._format_resolution_analysis(results)}

2. Descriptor Characteristics
---------------------------
Fourier Descriptor Statistics:
{self._format_descriptor_statistics(results['fourier'])}

Zernike Moment Statistics:
{self._format_descriptor_statistics(results['zernike'])}

3. Processing Efficiency
----------------------
{self._format_processing_analysis(results['processing_times'])}

4. Recommendations
----------------
Based on the analysis:
- Optimal resolution: {self._determine_optimal_resolution(results)}
- Most stable descriptor: {self._determine_stable_descriptor(results)}
- Best performance/accuracy trade-off: {self._analyze_tradeoffs(results)}
"""

        with open(output_dir / 'detailed_analysis_report.txt', 'w') as f:
            f.write(report)

    def _format_resolution_analysis(self, results: Dict) -> str:
        """Format resolution analysis results."""
        if not results['processing_times']:
            return "No resolution analysis data available."

        analysis = []
        for resolution in results['processing_times'].keys():
            analysis.append(f"Resolution {resolution} vertices:")
            analysis.append(f"- Processing time: {results['processing_times'][resolution]:.2f} seconds")
            if results['fourier'][resolution]:
                analysis.append(f"- Fourier descriptor variance: {np.var(results['fourier'][resolution]):.4f}")
            if results['zernike'][resolution]:
                analysis.append(f"- Zernike moment variance: {np.var(results['zernike'][resolution]):.4f}")
            analysis.append("")
        return "\n".join(analysis)

    def _format_descriptor_statistics(self, descriptor_results: Dict) -> str:
        """Format descriptor statistics."""
        if not descriptor_results:
            return "No descriptor statistics available."

        stats = []
        for resolution, data in descriptor_results.items():
            if data:  # Only process if we have data
                stats.append(f"Resolution {resolution}:")
                stats.append(f"- Mean magnitude: {np.mean(data):.4f}")
                stats.append(f"- Standard deviation: {np.std(data):.4f}")
                stats.append(f"- Range: {np.ptp(data):.4f}")
                stats.append("")
        return "\n".join(stats)

    def _determine_optimal_resolution(self, results: Dict) -> str:
        """Determine optimal resolution based on performance metrics."""
        if not results['processing_times']:
            return "Unable to determine optimal resolution due to lack of data"

        resolutions = list(results['processing_times'].keys())
        scores = []

        for resolution in resolutions:
            # Calculate a score based on processing time and descriptor stability
            time_score = 1 / results['processing_times'][resolution]

            # Only include descriptor stability if we have data
            fourier_stability = 1 / np.var(results['fourier'][resolution]) if results['fourier'][resolution] else 0
            zernike_stability = 1 / np.var(results['zernike'][resolution]) if results['zernike'][resolution] else 0

            total_score = time_score * 0.4 + fourier_stability * 0.3 + zernike_stability * 0.3
            scores.append(total_score)

        return f"{resolutions[np.argmax(scores)]} vertices"

    def _determine_stable_descriptor(self, results: Dict) -> str:
        """Determine which descriptor type is more stable across resolutions."""
        if not (results['fourier'] and results['zernike']):
            return "Unable to determine stable descriptor due to lack of data"

        fourier_variance = np.mean([np.var(data) for data in results['fourier'].values() if len(data) > 0])
        zernike_variance = np.mean([np.var(data) for data in results['zernike'].values() if len(data) > 0])

        return "Fourier" if fourier_variance < zernike_variance else "Zernike"

    def _analyze_tradeoffs(self, results: Dict) -> str:
        """Analyze performance/accuracy trade-offs."""
        if not results['processing_times']:
            return "Unable to analyze trade-offs due to lack of data"

        optimal_resolution = self._determine_optimal_resolution(results)
        return f"{optimal_resolution} provides the best balance between accuracy and performance"



    def _analyze_performance_metrics(self, results: Dict) -> Dict:
        """
        Analyze performance metrics for different resolutions.
        """
        performance_metrics = {}

        for resolution in results['processing_times'].keys():
            performance_metrics[resolution] = {
                'processing_time': results['processing_times'][resolution],
                'memory_usage': results.get('memory_usage', {}).get(resolution, 0)
            }

        return performance_metrics



    def run_analysis(self, test_models: List[Path],
                     resolutions: List[int] = [500, 1000, 2000],
                     output_dir: Path = Path('./output')) -> None:
        """
        Run the complete analysis process with enhanced metrics.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze models with different resolutions
        results = self.analyze_with_different_resolutions(test_models, resolutions)

        if not results['processing_times']:
            logging.error("No results were generated during analysis")
            return

        try:
            # Generate basic visualizations
            self.generate_comparative_visualizations(results, output_dir)

            # Generate enhanced analysis metrics
            self.generate_analysis_metrics(results, output_dir)

            # Generate detailed report
            self.generate_detailed_report(results, output_dir)

            logging.info(f"Analysis completed. Results saved to {output_dir}")

        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}")
            raise