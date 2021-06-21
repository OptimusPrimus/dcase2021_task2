import numpy as np

from common.data.datasets.dcase2021_task2 import MACHINE_TYPES, NUM_SECTIONS
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import hmean


def evaluate(df, machine_type_idx=None, section=None):

    if machine_type_idx == None:
        raise NotImplementedError()

    machine_type_idxs = [machine_type_idx] if (machine_type_idx is not None) else list(range(len(MACHINE_TYPES)))

    if section == None:
        sections = [0, 1, 2, 3, 4, 5]
    elif section == -1:
        sections = [0, 1, 2]
    elif 0 <= section < 6:
        sections = [section]
    else:
        raise AttributeError

    anomaly_score = df['anomaly_score'].to_numpy()
    section = df['section'].to_numpy()
    source = df['source'].to_numpy()
    anomaly = df['anomaly'].to_numpy()

    results = {}
    for m in machine_type_idxs:
        results[MACHINE_TYPES[m]] = dict()
        for s in sections:
            results[MACHINE_TYPES[m]][s] = dict()
            for d in [0, 1]:
                indices = (section == s) & (source == d) & (machine_type_idx == m)
                try:
                    aucs = roc_auc_score(anomaly[indices], anomaly_score[indices])
                    paucs = roc_auc_score(anomaly[indices], anomaly_score[indices], max_fpr=0.1)
                except:
                    aucs = 0
                    paucs = 0

                results[MACHINE_TYPES[m]][s]['source' if d == 1 else 'target'] = {
                    'auc': aucs,
                    'pauc': paucs
                }

    return results


def log_metrics(metrics, log_fun, id):
    source = []
    target = []

    for m in metrics:
        for s in metrics[m]:
            for d in metrics[m][s]:
                auc, pauc = metrics[m][s][d]['auc'], metrics[m][s][d]['pauc']
                if auc == 0 and pauc == 0:
                    continue
                if d == 'source':
                    source.append(auc)
                    source.append(pauc)
                else:
                    target.append(auc)
                    target.append(pauc)
                log_fun(f"{id}_auc_{m}_{s}_{d}", auc)
                log_fun(f"{id}_pauc_{m}_{s}_{d}", pauc)

    invalid = len(source) == 0 or len(target) == 0

    log_fun(f'{id}_auc_pauc_hmean', 0 if invalid else hmean(source + target))
    log_fun(f'{id}_auc_pauc_source_hmean', 0 if invalid else hmean(source))
    log_fun(f'{id}_auc_pauc_target_hmean', 0 if invalid else hmean(target))