from typing import List, Any
import os

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import pytorch_lightning as pl
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from common.architectures import ResNet
from common.data.datasets.dcase2021_task2 import MACHINE_TYPES, NUM_SECTIONS
from common.layers.normalization import Lambda, MeanStdNormalization
from experiments.utils import get_feature_extractor, get_mean_std

from experiments.evaluate import evaluate, log_metrics

import torch

from experiments.utils import stack_numpy, column_names, stack_torch


class ClassificationExperiment(pl.LightningModule):

    def __init__(
            self,
            da_task='none',
            margin=0.5,
            da_lambda=0.90,
            rampdown_length=60,
            rampdown_start=30,
            **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs['da_task'] = da_task
        self.kwargs['da_lambda'] = da_lambda
        self.kwargs['margin'] = margin
        self.kwargs['rampdown_length'] = rampdown_length
        self.kwargs['rampdown_start'] = rampdown_start
        self.save_hyperparameters()

        self.machine_type_idx = MACHINE_TYPES.index(kwargs['machine_type'])
        self.section = kwargs['section']

        self.debug = kwargs['debug']

        self.encoder = get_feature_extractor(**kwargs)

        normalize = kwargs['normalize']
        task = kwargs['experiment']

        if normalize in ['mean_std', 'mean_std_learnable']:
            requires_grad = normalize == 'mean_std_learnable'
            mean, std = get_mean_std(kwargs, cuda=not self.debug)
            self.normalize = MeanStdNormalization(mean, std, requires_grad=requires_grad)
            input_shape = (1, self.kwargs['n_mels'], 100)
        elif normalize in ['batchnorm']:
            self.normalize = torch.nn.BatchNorm2d(1)
            input_shape = (1, self.kwargs['n_mels'], 100)
        elif normalize in ['none']:
            self.normalize = Lambda(lambda x: x)
            input_shape = (1, self.kwargs['n_mels'], 100)
        else:
            raise AttributeError(f"Normalization '{normalize}' unknown.")

        if task == 'multi_section':
            assert self.section == None
            self.task = MultiSectionPredictionTask()
        else:
            raise AttributeError

        if self.kwargs['architecture'] == 'resnet':
            self.network = ResNet(
                input_shape=input_shape,
                n_classes=self.task.num_classes(),
                rho_f=kwargs['rho_f'],
                rho_t=kwargs['rho_t']
            )
        else:
            raise AttributeError

        if da_task == 'ccsa':
            self.da_task = CCSA(margin)
        elif da_task == 'none':
            self.da_task = None
        else:
            raise AttributeError(f'Task {da_task} unknown.')

    def forward(self, batch):
        batch['input'] = self.normalize(self.encoder(batch['audio']))
        self.network(batch)
        return batch

    def training_step(self, batch, batch_idx):
        if self.da_task is None:
            if self.kwargs['proxy_outliers'] != 'none' and len(batch) == 2:
                in_batch, out_batch = batch[0], batch[1]
                output = self({'audio': torch.cat([b['audio'] for b in [in_batch, out_batch]], dim=0)})
                for k in ['input', 'logits', 'embedding']:
                    in_batch[k] = output[k][:len(in_batch['audio'])]
                    out_batch[k] = output[k][len(in_batch['audio']):]
                loss = self.task(in_batch).mean()
                po_loss = self.task.forward_po(in_batch, out_batch).mean()
                self.log('train_loss', loss.item())
                self.log('train_po_loss', po_loss.item())
                loss += self.kwargs['proxy_outlier_lambda'] * po_loss
            else:
                batch = batch[0]
                batch = self(batch)
                loss = self.task(batch).mean()
                self.log('train_loss', loss.item())
        elif type(self.da_task) in [CCSA]:
            if len(batch) == 2:
                source, target = batch
                output = self({'audio': torch.cat([b['audio'] for b in [source, target]], dim=0)})
                for k in ['input', 'logits', 'embedding']:
                    source[k] = output[k][:len(source['audio'])]
                    target[k] = output[k][len(source['audio']):]

                concatenated = {k: stack_torch(k, [source, target]) for k in source}
                loss = self.task(concatenated).mean()
            else:
                source, target, po = batch
                output = self({'audio': torch.cat([b['audio'] for b in [source, target, po]], dim=0)})
                for k in ['input', 'logits', 'embedding']:
                    source[k] = output[k][:len(source['audio'])]
                    target[k] = output[k][len(source['audio']):len(source['audio']) + len(target['audio'])]
                    po[k] = output[k][len(source['audio']) + len(target['audio']):]
                concatenated = {k: stack_torch(k, [source, target]) for k in source}
                loss = self.task(concatenated).mean() + self.kwargs['proxy_outlier_lambda'] * self.task.forward_po(source, po).mean()

            self.log('train_loss', loss.item())

            if type(self.da_task) == CCSA:
                semantic_loss, contrastive_loss = self.da_task(source, target)
                self.log('train_da_semantic', semantic_loss.mean().item())
                self.log('train_da_contrastive', contrastive_loss.mean().item())
                da_loss = 0.5 * semantic_loss.mean() + 0.5 * contrastive_loss.mean()
            else:
                da_loss = self.da_task(source, target)

            self.log('train_da_loss', da_loss.item())
            loss = self.kwargs['da_lambda'] * loss + (1 - self.kwargs['da_lambda']) * da_loss
        else:
            raise AttributeError('Unknown DA type')

        return loss

    def validation_step(self, batch, batch_idx, val=True):
        id = 'val' if val else 'test'
        self(batch)
        loss = self.task(batch)
        self.log(f'{id}_loss', loss.mean().item())

        return {
            'logits': batch['logits'],
            'source': batch['source'],
            'anomaly': batch['anomaly'],
            'section': batch['section'],
            'machine_type_idx': torch.ones_like(batch['source']) * self.machine_type_idx,
            'train': batch['train'],
            'path': batch['path'],
            'part': batch['part'],
            f'{id}_loss': loss
        }

    def validation_epoch_end(self, outputs: List[Any], val=True) -> None:
        id = 'val' if val else 'test'

        outputs = {k: stack_numpy(k, outputs) for k in outputs[0]}
        df = pd.concat([pd.DataFrame(outputs[k], columns=column_names(k, outputs[k])) for k in outputs], axis=1)
        df['anomaly_score'] = self.task.anomaly_fun(df)

        if id == 'test':
            df.to_csv(os.path.join(self.kwargs['log_dir'], f'{id}_{self.current_epoch}.csv'))

        # average over snippets
        df = df.groupby('path').mean()
        # compute metrics
        metrics = evaluate(
            df[df['train'] == False],
            machine_type_idx=self.machine_type_idx,
            section=self.section
        )

        if self.current_epoch > 1 and hasattr(self.task, 'metrics'):
            for k, v in self.task.metrics(df).items():
                self.log(id + '_' + k, v)

        log_metrics(metrics, self.log, id)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, val=False)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        self.validation_epoch_end(outputs, val=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.kwargs['learning_rate'],
            weight_decay=self.kwargs['weight_decay']
        )

        def linear_rampdown(rampdown_length=self.kwargs['rampdown_length'], start=self.kwargs['rampdown_start'], last_value=self.kwargs['min_learning_rate']):
            def warpper(epoch):
                if epoch <= start:
                    return 1.
                elif epoch - start < rampdown_length:
                    return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
                else:
                    return last_value
            return warpper

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            linear_rampdown()
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    @staticmethod
    def add_model_specific_args(parent_parser, **defaults):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--architecture', type=str, default='resnet', choices=['resnet'])
        parser.add_argument('--normalize', type=str, default='batchnorm', choices=['mean_std', 'mean_std_learnable', 'batchnorm', 'none'])
        parser.add_argument('--rho_t', type=int, default=12)
        parser.add_argument('--rho_f', type=int, default=12)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--min_learning_rate', type=float, default=1e-5)
        parser.add_argument('--rampdown_length', type=int, default=60)
        parser.add_argument('--rampdown_start', type=int, default=30)
        parser.add_argument('--proxy_outlier_lambda', type=float, default=0.5)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--da_task', type=str, default='none', choices=['ccsa', 'none'])
        parser.add_argument('--margin', type=float, default=1.0)
        parser.add_argument('--da_lambda', type=float, default=0.9)
        parser.add_argument('--debug', dest='debug', action='store_true')
        parser.set_defaults(debug=False)

        return parser


class MultiSectionPredictionTask(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.uniform_dist = torch.nn.Parameter(torch.ones((1, NUM_SECTIONS)) / NUM_SECTIONS, requires_grad=False)

    def forward(self, batch):

        n_inliers = len(batch['audio'])
        device = batch['audio'].device
        targets = torch.zeros(
            n_inliers, NUM_SECTIONS,
            dtype=torch.float32,
            device=device
        )

        if batch.get('mix_factor') is None:
            targets[torch.arange(n_inliers), batch['section']] = 1
        else:
            targets[torch.arange(n_inliers), batch['section']] = batch['mix_factor'].float()
            targets[torch.arange(n_inliers), batch['other_section']] = 1 - batch['mix_factor'].float()
        loss = - (torch.log_softmax(batch['logits'], dim=1) * targets).sum(dim=1)

        return loss

    def forward_po(self, in_batch, out_batch):
        n_outliers = len(out_batch['audio'])
        return -(torch.log_softmax(out_batch['logits'], dim=1) * self.uniform_dist.repeat(n_outliers, 1)).sum(dim=1)

    @staticmethod
    def metrics(df):
        section_predicted = df[[f'logits_{i}' for i in range(NUM_SECTIONS)]].to_numpy()
        section = df['section'].to_numpy()

        return {
            f'prediction_acc': accuracy_score(section, section_predicted.argmax(axis=1))
        }

    @staticmethod
    def anomaly_fun(df):

        section_predicted = df[[f'logits_{i}' for i in range(NUM_SECTIONS)]].to_numpy()
        section = df['section']

        return 1 - softmax(section_predicted)[np.arange(len(section_predicted)), section]

    def num_classes(self):
        return NUM_SECTIONS


class CCSA(torch.nn.Module):

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, source_batch, target_batch):

        source = source_batch['embedding']
        target = target_batch['embedding']

        source_section = source_batch['section']
        target_section = target_batch['section']

        loss_s = []
        loss_c = []

        for e_s, s_s in zip(source, source_section):
            losses_semantic = []
            losses_contrastive = []
            for e_t, s_t in zip(target, target_section):
                if s_s == s_t:
                    losses_semantic.append(self.distance(e_s, e_t)**2)
                else:
                    losses_contrastive.append(torch.maximum(torch.tensor(0.0).to(e_s.device), self.margin - self.distance(e_s, e_t)) ** 2)

            if len(losses_semantic) == 0:
                loss_s.append(torch.tensor(0.0).to(e_s.device))
            else:
                loss_s.append(torch.stack(losses_semantic).sum() / len(target_section))

            if len(losses_contrastive) == 0:
                loss_c.append(torch.tensor(0.0).to(e_s.device))
            else:
                loss_c.append(torch.stack(losses_contrastive).sum() / len(target_section))

        return torch.stack(loss_s), torch.stack(loss_c)

    @staticmethod
    def distance(x, y):
        return torch.sqrt(((x - y) ** 2).mean())
