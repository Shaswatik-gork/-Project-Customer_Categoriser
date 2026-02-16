import sys
import os
from typing import Tuple
import numpy as np
from pandas import DataFrame

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)
from src.exception import CustomerException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data


class CustomerSegmentationModel:
    def __init__(self, preprocessing_object: object, clustering_model: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.clustering_model = clustering_model
        self.trained_model_object = trained_model_object

    def predict(self, X: DataFrame):
        try:
            transformed_feature = self.preprocessing_object.transform(X)
            clusters = self.clustering_model.predict(transformed_feature)
            transformed_with_cluster = np.c_[transformed_feature, clusters]
            return self.trained_model_object.predict(transformed_with_cluster)
        except Exception as e:
            raise CustomerException(e, sys) from e


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method")

        try:
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            train_clusters = kmeans.fit_predict(x_train)
            test_clusters = kmeans.predict(x_test)

            # Add cluster feature
            x_train_clustered = np.c_[x_train, train_clusters]
            x_test_clustered = np.c_[x_test, test_clusters]

            # Logistic Regression
            model = LogisticRegression(max_iter=1000)
            model.fit(x_train_clustered, y_train)

            # Evaluation
            y_pred = model.predict(x_test_clustered)

            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            preprocessing_obj = self.utils.load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )

            customer_segmentation_model = CustomerSegmentationModel(
                preprocessing_object=preprocessing_obj,
                clustering_model=kmeans,
                trained_model_object=model,
            )

            trained_model_path = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(trained_model_path, exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=customer_segmentation_model,
            )

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )

            logging.info("Model training completed successfully")

            return model_trainer_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e
