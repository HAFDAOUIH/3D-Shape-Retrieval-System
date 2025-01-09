from pathlib import Path
import logging
from src.pipeline.model_processor import ModelProcessor
from src.search_engine.model_search_engine import ModelSearchEngine

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Initialize components
    try:
        logger.info("Initializing components...")
        data_dir = Path("data/raw_models/3D Models/All Models")
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            return

        processor = ModelProcessor(data_dir)
        logger.info("ModelProcessor initialized successfully")

        search_engine = ModelSearchEngine(Path("data/search_index.pkl"))
        logger.info("ModelSearchEngine initialized successfully")
    except Exception as e:
        logger.error(f"Error during component initialization: {str(e)}")
        return

    # Process query model
    try:
        query_path = Path("data/raw_models/3D Models/All Models/5DSOM_fakej.obj")
        logger.info(f"Processing query model: {query_path}")
        query_result = processor.process_single_model(query_path)
        if query_result is None:
            logger.error("Failed to process query model")
            return
        logger.info("Query model processed successfully")
    except Exception as e:
        logger.error(f"Error processing query model: {str(e)}")
        return

    # Load or build index
    try:
        search_engine.load_index()
        logger.info("Loaded existing search index")
    except FileNotFoundError:
        logger.warning("Search index not found. Building a new one...")
        database_results = processor.process_database()
        search_engine.build_index(database_results)
        search_engine.save_index()
        logger.info("Search index built and saved successfully")
    except Exception as e:
        logger.error(f"Error loading or building search index: {str(e)}")
        return

    # Search similar models
    try:
        logger.info("Searching for similar models...")
        results = search_engine.search(
            query_result['fourier'],
            query_result['zernike'],
            top_k=10,
            weights=(0.6, 0.4)
        )
        logger.info("Search completed successfully")

        # Print results
        print("\nSearch Results:")
        for model_path, similarity in results:
            print(f"Model: {model_path}")
            print(f"Similarity Score: {similarity:.4f}")
            print("-" * 50)
    except Exception as e:
        logger.error(f"Error during model search: {str(e)}")
        return

if __name__ == "__main__":
    main()
