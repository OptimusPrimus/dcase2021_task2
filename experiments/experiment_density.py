from typing import List, Any
import os

import pandas as pd
from argparse import ArgumentParser
import pytorch_lightning as pl

import torch.nn.functional as F
from common.data.datasets.dcase2021_task2 import MACHINE_TYPES
from common.layers.normalization import Lambda, MeanStdNormalization
from experiments.utils import get_feature_extractor, get_mean_std, stack_numpy, column_names

from experiments.evaluate import evaluate, log_metrics

import torch

from common.architectures.maf import MADEMOG, MADE, MAF, MAFMOG, AE


class DensityEstimate(pl.LightningModule):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        self.save_hyperparameters()

        self.machine_type_idx = MACHINE_TYPES.index(kwargs['machine_type'])
        self.section = kwargs['section']

        self.debug = kwargs['debug']

        self.encoder = get_feature_extractor(**kwargs)

        # self.normalize_target = Lambda(lambda x: x)
        if kwargs['normalize'] in ['mean_std', 'mean_std_learnable']:
            mean, std = get_mean_std(kwargs, cuda=not self.debug)
            requires_grad = kwargs['normalize'] == 'mean_std_learnable'
            self.normalize = MeanStdNormalization(mean, std, requires_grad=requires_grad)
            self.normalize_target = MeanStdNormalization(mean, std, requires_grad=False)
        elif kwargs['normalize'] in ['none']:
            self.normalize = Lambda(lambda x: x)
            self.normalize_target = Lambda(lambda x: x)
        else:
            raise AttributeError(f"Normalization '{kwargs['normalize']}' unknown.")

        self.task = NegativeLogLikelihood(margin=self.kwargs['margin'])

        if self.kwargs['architecture'] in ['made', 'gmade']:
            assert self.kwargs['n_gaussians'] > 0

            if self.kwargs['n_gaussians'] == 1:
                self.network = MADE(
                    self.kwargs['n_mels']*5,
                    self.kwargs['hidden_size'],
                    self.kwargs['hidden_dims'],
                    cond_label_size=6
                )
            else:
                self.network = MADEMOG(
                    self.kwargs['n_gaussians'],
                    self.kwargs['n_mels']*5,
                    self.kwargs['hidden_size'],
                    self.kwargs['hidden_dims'],
                    cond_label_size=6
                )
        elif self.kwargs['architecture'] in ['maf']:
            assert self.kwargs['n_gaussians'] > 0

            if self.kwargs['n_gaussians'] == 1:
                self.network = MAF(
                    self.kwargs['hidden_dims'],
                    self.kwargs['n_mels']*5,
                    self.kwargs['hidden_size'],
                    1,
                    cond_label_size=6
                )
            else:
                self.network = MAFMOG(
                    self.kwargs['hidden_dims'],
                    self.kwargs['n_gaussians'],
                    self.kwargs['n_mels']*5,
                    self.kwargs['hidden_size'],
                    1,
                    cond_label_size=6
                )
        elif self.kwargs['architecture'] == 'ae':
            self.network = AE(
                self.kwargs['n_mels']*5,
                cond_label_size = 6
            )
        else:
            raise AttributeError(f"Architecture {self.kwargs['architecture']} unknown!")

    def forward(self, batch):
        spectorgram = self.encoder(batch['audio'])
        batch['input'] = self.normalize(spectorgram)
        batch['target'] = self.normalize_target(spectorgram)
        input = batch['input'].transpose(2, 3).reshape(len(batch['input']), -1)
        section = F.one_hot(batch['section'], num_classes=6).float()
        batch['logp'] = self.network.log_prob(input, section) / input.shape[1]
        return batch

    def training_step(self, batch, batch_idx):
        if self.kwargs['proxy_outliers'] != 'none':
            in_batch, out_batch = batch[0], batch[1]
            device = in_batch['section'].device
            if out_batch.get('section') is None:
                out_batch['section'] = torch.randint(6, in_batch['section'].shape, device=device)
            else:
                same_machine = out_batch['machine_type_idx'] == self.machine_type_idx
                out_batch['section'][same_machine] = (out_batch['section'][same_machine] + torch.randint(low=1, high=6, size=out_batch['section'][same_machine].shape, device=device)) % 6
                out_batch['section'][~same_machine] = torch.randint(6, in_batch['section'][~same_machine].shape, device=device)

            output = self({
                'audio': torch.cat([b['audio'] for b in [in_batch, out_batch]], dim=0),
                'section': torch.cat([b['section'] for b in [in_batch, out_batch]], dim=0)
            })
            for k in ['input', 'logp', 'target']:
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

        return loss

    def validation_step(self, batch, batch_idx, val=True):
        # assumes every batch contains only one element
        id = 'val' if val else 'test'
        self(batch)
        negative_log_likelihood = self.task(batch)
        self.log(f'{id}_loss', negative_log_likelihood.mean().item())

        return {
            'neg_logp': negative_log_likelihood,
            'source': batch['source'],
            'anomaly': batch['anomaly'],
            'section': batch['section'],
            'machine_type_idx': torch.ones_like(batch['source']) * self.machine_type_idx,
            'train': batch['train'],
            'path': batch['path'],
            'part': batch['part'],
            f'{id}_loss': negative_log_likelihood
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

        def linear_rampdown(rampdown_length=60, start=30, last_value=self.kwargs['min_learning_rate']):
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

        parser.add_argument('--architecture', type=str, default='ae', choices=['made', 'gmade', 'maf', 'ae'])

        parser.add_argument('--n_gaussians', type=int, default=1)
        parser.add_argument('--hidden_dims', type=int, default=4)
        parser.add_argument('--hidden_size', type=int, default=2048)
        parser.add_argument('--normalize', type=str, default='mean_std', choices=['mean_std'])
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--min_learning_rate', type=float, default=1e-5)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--margin', type=float, default=0.0)
        parser.add_argument('--proxy_outlier_lambda', type=float, default=0.5)
        parser.add_argument('--debug', dest='debug', action='store_true')
        parser.set_defaults(debug=False)

        return parser


class NegativeLogLikelihood(torch.nn.Module):

    def __init__(self, margin=0.0):
        super().__init__()
        self.mrl = torch.nn.MarginRankingLoss(margin=margin)

    def forward(self, batch):
        return -batch['logp']

    def forward_po(self, in_batch, out_batch):
        p = in_batch['logp']
        n = out_batch['logp']
        min_length = min(len(n), len(p))
        n = n[:min_length]
        p = p[:min_length]

        return self.mrl(p, n, torch.ones(min_length).to(n.device))

    @staticmethod
    def anomaly_fun(df):
        return df[f'neg_logp'].to_numpy()

    def num_classes(self):
        return self.n_sections

