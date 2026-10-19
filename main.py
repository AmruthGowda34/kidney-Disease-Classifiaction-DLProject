from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline
)
from cnnClassifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline
)
from cnnClassifier.pipeline.stage_03_model_training import (
    ModelTrainingPipeline
)
from cnnClassifier.pipeline.stage_04_model_evaluation import (
    ModelEvaluationPipeline
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


STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    obj = ModelEvaluationPipeline()
    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise
