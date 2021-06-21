import numpy as np
import os
import pandas as pd
import scipy
from common.data.datasets.dcase2021_task2 import MACHINE_TYPES
from experiments.evaluate import evaluate
from scipy.stats import hmean
LOG_DIR = os.path.join('..', 'logs')
OUT_DIR = 'submissions'


def aggregate_fun(df, machine_type):
    return df.groupby('path').mean()


def load_data_frame(model_ids, aggregate_fun=aggregate_fun):

    all_samples = []
    for d in model_ids:
        for m in model_ids[d]:
            for s in [0, 1, 2, 3, 4, 5]:
                scores = []
                for id in model_ids[d][m]:
                    if os.path.exists(os.path.join(LOG_DIR, id, 'test_99.csv')):
                        path = os.path.join(LOG_DIR, id, 'test_99.csv')
                    else:
                        path = os.path.join(LOG_DIR, id, 'test_2.csv')
                    df = pd.read_csv(path)
                    df['path'] = df['path'].apply(os.path.basename)

                    df = aggregate_fun(df, m)
                    df = df.loc[(df['machine_type_idx'] == MACHINE_TYPES.index(m)) & (df['section'] == s)]

                    # get mean / std
                    mean = df.loc[(df['train'] == True), 'anomaly_score'].to_numpy().mean()
                    std = df.loc[(df['train'] == True), 'anomaly_score'].to_numpy().std()
                    # normalize
                    # df['anomaly_score'] = (df['anomaly_score'].to_numpy() - mean) / std
                    scores.append((df['anomaly_score'] - mean) / std)

                averaged = pd.concat(scores, axis=1).mean(axis=1)
                df = df.copy()
                df['anomaly_score'] = averaged
                all_samples.append(df.loc[(df['source'] == (d == 'source'))].copy())

    all_samples = pd.concat(all_samples, axis=0)
    assert not all_samples.isnull().values.any()

    return all_samples


def aggregate_metrics(df):
    aggregated = []
    for i in range(7):
        metrics = evaluate(df.loc[(df['train'] == False) & (df['machine_type_idx'] == i)], machine_type_idx=i,section=-1)

        all = []

        for m in metrics:
            for s in [0, 1, 2]:  # metrics[m]:
                for d in metrics[m][s]:
                    all.append(metrics[m][s][d]['auc'])
                    all.append(metrics[m][s][d]['pauc'])

        aggregated.append(hmean(all))
    return aggregated


def aggregate_metrics_yaml(df):
    aggregated = []

    results = {}

    for i in range(7):
        metrics = evaluate(df.loc[(df['train'] == False) & (df['machine_type_idx'] == i)], machine_type_idx=i,section=-1)

        aucs = []
        paucs = []
        m = MACHINE_TYPES[i]

        for s in [0, 1, 2]:  # metrics[m]:
            for d in metrics[m][s]:
                aucs.append(metrics[m][s][d]['auc'])
                paucs.append(metrics[m][s][d]['pauc'])

        results[m] = hmean(aucs), hmean(paucs)
    return aggregated
