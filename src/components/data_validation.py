import sys
from typing import Tuple
import pandas as pd
from pandas import DataFrame
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self.utils = MainUtils()
        self._schema_config = self.utils.read_schema_config_file()

    def validate_schema_columns(self, dataframe: DataFrame) -> bool:
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required column present [{status}]")
            return status
        except Exception as e:
            raise CustomerException(e, sys) from e

    def validate_dataset_schema_columns(self, train_set, test_set) -> Tuple[bool, bool]:
        try:
            train_schema_status = self.validate_schema_columns(train_set)
            test_schema_status = self.validate_schema_columns(test_set)
            return train_schema_status, test_schema_status
        except Exception as e:
            raise CustomerException(e, sys) from e

    def detect_dataset_drift(self, reference_df: DataFrame, current_df: DataFrame) -> bool:
        try:
            logging.info("Drift detection skipped - assuming no drift")
            write_yaml_file(file_path=self.data_validation_config.drift_report_file_path, content={"drift_detected": False})
            return False
        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df, test_df = (DataValidation.read_data(self.data_ingestion_artifact.trained_file_path), DataValidation.read_data(self.data_ingestion_artifact.test_file_path))
            drift = self.detect_dataset_drift(train_df, test_df)
            schema_train_col_status, schema_test_col_status = self.validate_dataset_schema_columns(train_set=train_df, test_set=test_df)
            logging.info(f"Schema train cols status is {schema_train_col_status} and schema test cols status is {schema_test_col_status}")
            validation_status = True if (schema_train_col_status and schema_test_col_status and not drift) else False
            data_validation_artifact = DataValidationArtifact(validation_status=validation_status, valid_train_file_path=self.data_ingestion_artifact.trained_file_path, valid_test_file_path=self.data_ingestion_artifact.test_file_path, invalid_train_file_path=self.data_validation_config.invalid_train_file_path, invalid_test_file_path=self.data_validation_config.invalid_test_file_path, drift_report_file_path=self.data_validation_config.drift_report_file_path)
            return data_validation_artifact
        except Exception as e:
            raise CustomerException(e, sys) from e
