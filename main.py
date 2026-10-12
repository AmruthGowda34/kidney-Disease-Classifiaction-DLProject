
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline
)
from cnnClassifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline
)

# Stage 1: Data Ingestion
STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

except Exception:
    logger.exception(f"Error in {STAGE_NAME}")
    raise


# Stage 2: Prepare Base Model
STAGE_NAME = "Prepare base model"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

except Exception:
    logger.exception(f"Error in {STAGE_NAME}")
    raise
