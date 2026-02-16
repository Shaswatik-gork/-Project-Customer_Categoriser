import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_clustering import CreateClusters
from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import SimpleImputerConfig
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_tranasformation_config: DataTransformationConfig):

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_tranasformation_config
        self.data_ingestion = DataIngestion()
        self.imputer_config = SimpleImputerConfig()
        self.utils = MainUtils()

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def prepare_data(self, train_set: DataFrame, test_set: DataFrame):
        """
        Rename columns to match the pipeline expectations.
        Data is already pre-processed in MongoDB so we just rename columns.
        """
        rename_map = {
            'Marital_Status': 'Marital Status',
            'Parental_Status': 'Parental Status',
            'Discount_Purchases': 'Discount Purchases',
            'Total_Promo': 'Total Promo'
        }
        train_set = train_set.rename(columns=rename_map)
        test_set = test_set.rename(columns=rename_map)

        # Select only required columns
        required_cols = [
            "Age", "Education", "Marital Status", "Parental Status",
            "Children", "Income", "Total_Spending", "Days_as_Customer",
            "Recency", "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold",
            "Web", "Catalog", "Store", "Discount Purchases", "Total Promo",
            "NumWebVisitsMonth"
        ]
        train_set = train_set[required_cols]
        test_set = test_set[required_cols]

        logging.info("Data prepared successfully.")
        return train_set, test_set

    def transform_data(self, train_set: DataFrame, test_set: DataFrame) -> DataFrame:
        logging.info("Entered transform_data method of DataTransformation class")

        try:
            numeric_features = [feature for feature in train_set.columns if train_set[feature].dtype != 'O']

            outlier_features = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending"]
            numeric_features = [x for x in numeric_features if x not in outlier_features]

            logging.info("Initialized StandardScaler, SimpleImputer")

            numeric_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                ("StandardScaler", StandardScaler())
            ])

            outlier_features_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(**self.imputer_config.__dict__)),
                ("transformer", PowerTransformer(standardize=True))
            ])

            preprocessor = ColumnTransformer([
                ("numeric pipeline", numeric_pipeline, numeric_features),
                ("Outliers Features Pipeline", outlier_features_pipeline, outlier_features)
            ])

            preprocessed_train_set = preprocessor.fit_transform(train_set)
            preprocessed_test_set = preprocessor.transform(test_set)

            columns = train_set.columns
            preprocessed_train_set = pd.DataFrame(preprocessed_train_set, columns=columns)
            preprocessed_test_set = pd.DataFrame(preprocessed_test_set, columns=columns)

            preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)
            os.makedirs(preprocessor_obj_dir, exist_ok=True)
            self.utils.save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            logging.info("Saved Preprocessor object to {}".format(preprocessor_obj_dir))

            return preprocessed_train_set, preprocessed_test_set

        except Exception as e:
            raise CustomerException(e, sys) from e

    def initiate_data_transformation(self):
        logging.info("Entered initiate_data_transformation method of Data_Transformation class")

        try:
            if self.data_validation_artifact.validation_status:
                train_set = DataTransformation.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_set = DataTransformation.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Skip get_new_features - data already pre-processed in MongoDB
                train_set, test_set = self.prepare_data(train_set, test_set)

                logging.info("Got the preprocessor object")

                preprocessed_train_set, preprocessed_test_set = self.transform_data(train_set, test_set)

                cluster_creator = CreateClusters()

                labelled_train_set = cluster_creator.initialize_clustering(preprocessed_data=preprocessed_train_set)
                labelled_test_set = cluster_creator.initialize_clustering(preprocessed_data=preprocessed_test_set)

                X_train = labelled_train_set.drop(columns=[TARGET_COLUMN])
                y_train = labelled_train_set[TARGET_COLUMN]

                X_test = labelled_test_set.drop(columns=[TARGET_COLUMN])
                y_test = labelled_test_set[TARGET_COLUMN]

                train_arr = np.c_[np.array(X_train), np.array(y_train)]
                test_arr = np.c_[np.array(X_test), np.array(y_test)]

                self.utils.save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                self.utils.save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )

                return data_transformation_artifact

            else:
                raise Exception("Data Validation Failed.")

        except Exception as e:
            raise CustomerException(e, sys) from e
