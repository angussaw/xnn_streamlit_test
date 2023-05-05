import logging
import mlflow
import os
import pandas as pd
from pathlib import Path
import tensorflow as tf
import xray_classifier as xray_clf

logger = logging.getLogger(__name__)

def train_pipeline(config: dict,
                   mode: str,
                   training_dataset: tf.data.Dataset,
                   validation_dataset: tf.data.Dataset,
                   test_dataset: tf.data.Dataset):
    """
    Performs training of a new model or fine tuning of an existing model
    depending on the mode ("training" or "fine-tuning")

    Evaluates model on test set, generating performance metrics and visualizations

    Model artifacts, evaluation metrics and visualizations are logged to MLFlow local server

    Args:
        config (dict):  training configurations as specified in a yaml file
        mode (str): "training" or "fine-tuning"
        training_dataset (tf.data.Dataset): dataset containing batches of images used for training
        validation_dataset (tf.data.Dataset): dataset containing batches of images used for validation
        test_dataset (tf.data.Dataset): dataset containing batches of images used for testing
    """

    model_params = config["model"]["model_params"]
    training_params = config["model"]["training_params"]
    compile_params = config["model"]["compile_params"]
    fine_tuning_params = config["fine_tuning"]

    logger.info("Intialising MLFlow...")
    artifact_name, description_str = xray_clf.utils.init_mlflow(config["mlflow"])

    if mode == "training":
        logger.info("Building Model...")
        model = xray_clf.modeling.model.build_model(**model_params)

    elif mode == "fine-tuning":
        logger.info("Retrieving Model...")
        model = xray_clf.modeling.model.retrieve_model(**fine_tuning_params["retrieve_params"])
        model.trainable=True
        for layer in model.layers[:fine_tuning_params["fine_tune_at"]]:
            layer.trainable = False
        bn_layers = list(filter(lambda x: isinstance(x, tf.keras.layers.BatchNormalization), model.layers))
        for layer in bn_layers:
            layer.trainable = False

    model.compile(loss=compile_params["loss_function"],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=compile_params["learning_rate"]),
                  metrics=config["metrics"])
    

    logger.info(f'Model {mode}...')
    history = model.fit(training_dataset,
                        epochs=training_params["n_epochs"],
                        validation_data=validation_dataset)
    
    logger.info("Evaluating model on test set...")
    evaluator = xray_clf.modeling.evaluation.Evaluator(model=model)
    final_metrics, test_metrics, visualizations_save_dir = evaluator.evaluate_model(metrics=config["metrics"],
                                                                                    test_data = test_dataset,
                                                                                    history=pd.DataFrame(history.history))

    with mlflow.start_run(
        run_name=config["mlflow"]["run_name"], description=description_str) as run:
        logger.info("Starting MLFlow Run...")

        logger.info("Saving model and params...")
        save_dir = os.path.dirname(visualizations_save_dir)
        model_dir = Path(os.sep.join([save_dir, "model"]))
        model_dir.mkdir()

        model_file_name = config["mlflow"]["model_name"]
        model_save_path = f'{model_dir}\{model_file_name}'
        model.save(model_save_path)
        mlflow.log_artifact(model_save_path)
        mlflow.log_params(compile_params)
        mlflow.log_params(training_params)

        logger.info("Saving metrics and model performance...")
        mlflow.log_metrics(final_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_artifacts(visualizations_save_dir)
        graph_uri = f"runs:/{run.info.run_id}/graph"
        logger.info(f"Model performance visualisation available at {graph_uri}")
        mlflow.end_run()

