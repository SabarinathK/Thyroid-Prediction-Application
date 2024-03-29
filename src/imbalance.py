import pandas as pd
import yaml
import os
import argparse
from imblearn.over_sampling import RandomOverSampler

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def balance(config_path):
    config = read_params(config_path)
    train_class_path=config["balanced_data"]["train_class"]
    train_label_path=config["balanced_data"]["train_label"]
    test_class_path=config["balanced_data"]["test_class"]
    test_label_path=config["balanced_data"]["test_label"]
    train_processed_path =config["processed"]["train_path"]
    test_processed_path =config["processed"]["test_path"]
    
    train_data=pd.read_csv(train_processed_path)
    test_data=pd.read_csv(test_processed_path)
    
    train_class=train_data["Class"].copy()
    train_label=train_data.drop('Class',axis=1).copy()

    test_class=test_data["Class"].copy()
    test_label=test_data.drop('Class',axis=1).copy()
    test_label.to_csv(test_label_path,index=False)
    test_class.to_csv(test_class_path,index=False)

    ros=RandomOverSampler(sampling_strategy='all')
    X_train_res,y_train_res=ros.fit_resample(train_label,train_class)
    
    X_train_res.to_csv(train_label_path,index=False)
    y_train_res.to_csv(train_class_path,index=False)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = balance(config_path=parsed_args.config)