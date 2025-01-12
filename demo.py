from pathlib import Path
import logging
from src.pipeline.model_processor import ModelProcessor
from src.search_engine.model_search_engine import ModelSearchEngine
from src.analysis.descriptor_analyzer import DescriptorAnalyzer
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        logger.info("Initializing components...")
        data_dir = Path("data/raw_models/3D Models/All Models")
        processor = ModelProcessor(data_dir)
        search_engine = ModelSearchEngine(Path("data/search_index.pkl"))

        # Load or build index
        try:
            search_engine.load_index()
            logger.info("Loaded existing search index")
        except FileNotFoundError:
            logger.warning("Search index not found. Building new one...")
            database_results = processor.process_database()
            search_engine.build_index(database_results)
            search_engine.save_index()
            logger.info("Search index built and saved")

        # Original analysis
        analyzer = DescriptorAnalyzer(processor, search_engine)

        # Run search demo
        query_path = data_dir / "5DSOM_fakej.obj"
        logger.info(f"Processing query model: {query_path}")
        query_result = processor.process_single_model(query_path)

        if query_result:
            results = search_engine.search(
                query_result['fourier'],
                query_result['zernike'],
                top_k=10,
                weights=(0.6, 0.4)
            )

            print("\nSearch Results:")
            for model_path, similarity in results:
                print(f"Model: {model_path}")
                print(f"Similarity Score: {similarity:.4f}")
                print("-" * 50)

        # Run basic analysis
        test_models = list(data_dir.glob("*.obj"))[:5]
        logger.info("Running basic descriptor analysis...")

        # Create results directory
        results_dir = Path("results/basic_analysis")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Run analysis (this will generate visualizations and report)
        analyzer.run_analysis(test_models, output_dir=results_dir)

        # Enhanced analysis
        logger.info("Running enhanced analysis...")
        enhanced_test_models = list(data_dir.glob("*.obj"))[:10]  # Using more models for enhanced analysis

        # Create enhanced results directory
        enhanced_results_dir = Path("results/enhanced_analysis")
        enhanced_results_dir.mkdir(parents=True, exist_ok=True)

        # Run enhanced analysis with different resolutions
        enhanced_results = analyzer.analyze_with_different_resolutions(
            enhanced_test_models,
            resolutions=[500, 1000, 2000]
        )

        # Generate enhanced results
        analyzer.generate_comparative_visualizations(enhanced_results, enhanced_results_dir)
        analyzer.generate_detailed_report(enhanced_results, enhanced_results_dir)

        logger.info(f"Basic analysis results saved to {results_dir}")
        logger.info(f"Enhanced analysis results saved to {enhanced_results_dir}")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()