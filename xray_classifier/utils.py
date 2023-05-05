"""Utils.py contains the general functions that will be used in during the end-to-end
 pipeline of X-ray image classifier
"""
from contextlib import contextmanager
import hashlib
import logging
import logging.config
import mlflow
import os
import time
from typing import List, Tuple
import yaml

logger = logging.getLogger(__name__)

@contextmanager
def timer(task: str = "Task"):
    """
    Logs how much time a code block takes

    Args:
        task (str, optional): Name of task, for logging purposes. Defaults to "Task".

    Example:

        with timer("showing example"):
            examplefunction()
    """
    start_time = time.time()
    yield
    logger.info(f"{task} completed in {time.time() - start_time:.5} seconds ---")


def setup_logging(
    logging_config_path="./conf/logging.yaml", default_level=logging.INFO
):
    """
    Set up configuration for logging utilities.

    Args:
        logging_config_path (str, optional): Path to YAML file containing configuration for
                                             Python logger. Defaults to "./conf/base/logging.yml".
        default_level (_type_, optional): logging object. Defaults to logging.INFO.
    """
    try:
        with open(logging_config_path, "rt") as file:
            log_config = yaml.safe_load(file.read())
            logging.config.dictConfig(log_config)

    except Exception as error:
        print("yoyoyo")
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")


def init_mlflow(mlflow_config: dict) -> Tuple[str, str]:
    """initialises mlflow parameters - tracking URI and experiment name.

    Takes in a configuration dictionary and sets the tracking URI
    and MLFlow experiment name. Returns the artifact name and the
    mlflow run description.

    Args:
        mlflow_config (dict): A dictionary containing the configurations
            of the mlflow run.

    Returns:
        artifact_name (str): Name of the artifact which the resultant
            trained model will be saved as. If none specified, the file
            will be saved as a hashed datetime.

        description (str): Description of the mlflow run, if any.
    """

    mlflow_tracking_uri =  os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info("Logging to MLFlow at %s", mlflow_tracking_uri)

    mlflow_experiment_name = mlflow_config["experiment_name"]
    mlflow.set_experiment(mlflow_experiment_name)
    logger.info("Logging to MLFlow Experiment: %s", mlflow_experiment_name)

    if mlflow_config["artifact_name"]:
        artifact_name = mlflow_config["artifact_name"]
    else:
        hashlib.sha1().update(str(time.time()).encode("utf-8"))
        artifact_name = hashlib.sha1().hexdigest()[:15]
    return artifact_name, mlflow_config.get("description", "")


    