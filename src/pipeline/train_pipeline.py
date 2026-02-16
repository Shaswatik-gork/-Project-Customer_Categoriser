import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_pusher import ModelPusher

from src.exception import CustomerException
from src.logger import logging

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelTrainerArtifact,
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_pusher_config = ModelPusherConfig()

    # ---------------------------
    # DATA INGESTION
    # ---------------------------
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            return data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise CustomerException(e, sys) from e

    # ---------------------------
    # DATA VALIDATION
    # ---------------------------
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            return data_validation.initiate_data_validation()

        except Exception as e:
            raise CustomerException(e, sys) from e

    # ---------------------------
    # DATA TRANSFORMATION
    # ---------------------------
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:

        try:
            logging.info("Starting data transformation")

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_tranasformation_config=self.data_transformation_config,
            )

            return data_transformation.initiate_data_transformation()

        except Exception as e:
            raise CustomerException(e, sys) from e

    # ---------------------------
    # MODEL TRAINING
    # ---------------------------
    def start_model_trainer(
        self,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:

        try:
            logging.info("Starting model training")

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            return model_trainer.initiate_model_trainer()

        except Exception as e:
            raise CustomerException(e, sys) from e

    # ---------------------------
    # MODEL PUSHER
    # ---------------------------
    def start_model_pusher(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            logging.info("Starting model push")

            model_pusher = ModelPusher(
                model_trainer_artifact=model_trainer_artifact,
                model_pusher_config=self.model_pusher_config,
            )

            return model_pusher.initiate_model_pusher()

        except Exception as e:
            raise CustomerException(e, sys) from e

    # ---------------------------
    # RUN PIPELINE
    # ---------------------------
    def run_pipeline(self) -> None:
        logging.info("Pipeline execution started")

        try:
            # Step 1
            data_ingestion_artifact = self.start_data_ingestion()

            # Step 2
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # Step 3
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )

            # Step 4
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )

            logging.info("Model trained successfully")

            # Step 5 (No Evaluation — Custom Architecture)
            self.start_model_pusher(
                model_trainer_artifact=model_trainer_artifact
            )

            logging.info("Model pushed successfully")
            logging.info("Pipeline execution completed successfully")

        except Exception as e:
            raise CustomerException(e, sys) from e


# -------------------------------------
# ENTRY POINT
# -------------------------------------
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
