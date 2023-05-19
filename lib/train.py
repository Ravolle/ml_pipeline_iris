import json
import os
import pickle
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.metrics import classification_report

mlflow.set_tracking_uri('http://158.160.11.51:90/')
mlflow.set_experiment('zhalyalovrr')

RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

METRICS = {
    'recall': partial(recall_score, average='macro'),
    'precision': partial(precision_score, average='macro'),
    'accuracy': accuracy_score,
}

LIST_OF_MODELS = {
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
}

def save_dict(data: dict, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)


def load_dict(filename: str):
    with open(filename, 'r') as f:
        return json.load(f)


def train_model(x, y, model_name):
    model = LIST_OF_MODELS[model_name]
    model.fit(x, y)
    return model


def train():
    with open('params.yaml', 'rb') as f:
        params_data = yaml.safe_load(f)

    config = params_data['train']
    task_dir = 'data/train'

    data = load_dict('data/predobrabotka_features/data.json')
    model = train_model(data['train_x'], data['train_y'], config['model'])

    preds = model.predict(data['train_x'])

    metrics = {}
    for metric_name in params_data['eval']['metrics']:
        metrics[metric_name] = METRICS[metric_name](data['train_y'], preds)

    cls_report = classification_report(data['train_y'], preds, output_dict=True)

    # save_data = {
    #     'train_x': train_x,
    #     'test_x': test_x,
    #     'train_y': train_y,
    #     'test_y': test_y,
    # }

    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # save_dict(save_data, os.path.join(task_dir, 'data.json'))
    save_dict(metrics, os.path.join(task_dir, 'metrics.json'))

    sns.heatmap(pd.DataFrame(data['train_x']).corr())

    plt.savefig('data/train/heatmap.png')

    with open('data/train/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    params = {}
    for i in params_data.values():
        params.update(i)

    params['run_type'] = 'train'

    print(f'train params - {params}')
    print(f'train metrics - {metrics}')

    # signature = infer_signature(x, preds)
    local_dir_png = "/home/student/ml_pipeline_iris/data/train/heatmap.png"
    local_dir_json = "/home/student/ml_pipeline_iris/data/train/cls_report.json"

    mlflow.sklearn.log_model(model, 'model.pkl')
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(local_dir_png)
    mlflow.log_artifacts(local_dir_json)


if __name__ == '__main__':
    train()
