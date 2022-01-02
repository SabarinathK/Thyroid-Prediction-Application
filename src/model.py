import pandas as pd
import yaml
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
import json

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def train_test(config_path):
    config = read_params(config_path)
    train_data_processed_path =config["preprocessed"]["train_path"]
    test_data_processed_path =config["preprocessed"]["test_path"]
    train_class_path =config["train_test"]["train_class"]
    train_label_path =config["train_test"]["train_label"]
    test_class_path =config["train_test"]["test_class"]
    test_label_path =config["train_test"]["test_label"]


    train_data=pd.read_csv(train_data_processed_path)
    test_data=pd.read_csv(test_data_processed_path)
    
    train_class=train_data["Class"].copy()
    train_label=train_data.drop('Class',axis=1).copy()
    train_class.to_csv(train_class_path,index=False)
    train_label.to_csv(train_label_path,index=False)

    test_class=test_data["Class"].copy()
    test_label=test_data.drop('Class',axis=1).copy()
    test_class.to_csv(test_class_path,index=False)
    test_label.to_csv(test_label_path,index=False)
    
    model=RandomForestClassifier( n_estimators=150,
                               criterion='gini',
                               max_depth=10)
    model.fit(train_label,train_class)

    from sklearn.metrics import classification_report

    y_pred=model.predict(test_label)
    cf_matrix=classification_report(test_class,y_pred)
    json.dump(cf_matrix)
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = train_test(config_path=parsed_args.config)