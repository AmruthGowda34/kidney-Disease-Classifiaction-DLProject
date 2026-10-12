
from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline
)

STAGE_NAME = "Data Ingestion stage"

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n"
            "x==========x"
        )

    except Exception:
        logger.exception("Data ingestion stage failed")
        raise
