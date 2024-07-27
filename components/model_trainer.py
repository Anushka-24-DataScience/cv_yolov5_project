import os
import sys
import yaml
from cv_yolov5_project.utils.main_utils import read_yaml_file
from cv_yolov5_project.logger import logging
from cv_yolov5_project.exception import AppException
from cv_yolov5_project.entity.config_entity import ModelTrainerConfig
from cv_yolov5_project.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")
            os.system("unzip data.zip")
            os.system("rm data.zip")

            # Rename data.yaml.yaml to data.yaml if it exists
            if os.path.exists("data.yaml.yaml"):
                os.rename("data.yaml.yaml", "data.yaml")

            # Ensure the data.yaml file is present
            if not os.path.exists("data.yaml"):
                raise AppException("data.yaml file not found after unzipping", sys)

            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)

            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")
            config['nc'] = int(num_classes)

            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            # Execute the training command
            os.system(f"cd yolov5/ && python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./models/custom_{model_config_file_name}.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results --cache")

            # Check if the training output file exists
            best_model_path = "yolov5/runs/train/yolov5s_results/weights/best.pt"
            if not os.path.exists(best_model_path):
                raise AppException(f"Training failed, best model file not found at {best_model_path}", sys)

            # Copy the trained model to the desired location
            os.system(f"cp {best_model_path} yolov5/")
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp {best_model_path} {self.model_trainer_config.model_trainer_dir}/")

            # Cleanup
            os.system("rm -rf yolov5/runs")
            os.system("rm -rf train")
            os.system("rm -rf valid")
            os.system("rm -rf data.yaml")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise AppException(e, sys)