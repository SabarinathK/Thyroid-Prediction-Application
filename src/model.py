import pandas as pd
import yaml
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import json

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def train_test(config_path):
    config = read_params(config_path)
    train_class_path =config["processed"]["train_class"]
    train_label_path =config["processed"]["train_label"]
    test_class_path =config["processed"]["test_class"]
    test_label_path =config["processed"]["test_label"]
    report_path=config["metrics"]["report"]

    train_label=pd.read_csv(train_label_path)
    train_class=pd.read_csv(train_class_path)
    test_class=pd.read_csv(test_class_path)
    test_label=pd.read_csv(test_label_path)


    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        model=RandomForestClassifier(n_estimators=150,
                                criterion='gini',
                                max_depth=10)

        model.fit(train_label,train_class)
        
    ###################################################################################################################################

        # METRICS___ Classification_report

        y_pred=model.predict(test_label)

        cl_report = pd.DataFrame(classification_report(test_class, y_pred , output_dict=True)).transpose()
        cl_report.to_csv(report_path, index= True)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = train_test(config_path=parsed_args.config)