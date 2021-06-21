import os
import sys
import git
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import socket
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from experiments.utils import get_logging_uri, get_data_module, get_best_model_path, get_feature_extractor, \
    get_experiment, get_logging_root

if __name__ == "__main__":
    parser = ArgumentParser()

    # program level args
    parser.add_argument('experiment', type=str)
    parser.add_argument('--version', type=str, default='debug')
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--data_module', type=str, default='mcm')
    parser.add_argument('--feature_extractor', type=str, default='mel_spectrogram')
    parser.add_argument('--data_root', type=str, default=os.path.join(os.path.expanduser('~'), 'shared', 'DCASE2021', 'task2'))
    parser.add_argument('--log_root', type=str, default=get_logging_root())

    # args for trainer
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.set_defaults(target_batch_only=False)

    # parse top level arguments
    args, _ = parser.parse_known_args()
    dict_args = vars(args)

    # args for experiment
    experiment_class, defaults = get_experiment(**dict_args)
    data_module_class = get_data_module(**dict_args)
    feature_extractor = get_feature_extractor(**dict_args)

    parser = experiment_class.add_model_specific_args(parser, **defaults)
    parser = data_module_class.add_model_specific_args(parser, **defaults)
    parser = feature_extractor.add_model_specific_args(parser, **defaults)

    # parse args
    args = parser.parse_args()
    dict_args = vars(args)

    # create data module
    data_module = data_module_class(**dict_args)
    dict_args['git_revision'] = str(git.Repo(search_parent_directories=True).head.object.hexsha)
    dict_args['cmd_command'] = ' '.join(sys.argv[1:])
    dict_args['hostname'] = socket.gethostname()

    logger = MLFlowLogger(
        experiment_name=dict_args['experiment'] + '_' + dict_args['version'],
        tracking_uri=get_logging_uri(**dict_args)
    )

    ##
    # Load pre-trained model (optional)
    ##

    if dict_args['run_id'] is not None:
        # save previous ID and path
        dict_args['prev_run_id'] = dict_args['run_id']
        dict_args['prev_log_dir'] = os.path.join(dict_args['log_root'], dict_args['prev_run_id'])
        # save current ID and path
        dict_args['run_id'] = logger.run_id
        dict_args['log_dir'] = os.path.join(dict_args['log_root'], dict_args['run_id'])
        # load previous experiments
        experiment = experiment_class.load_from_checkpoint(
            get_best_model_path(dict_args['prev_log_dir']),
            data_root=dict_args['data_root'],
            log_root=dict_args['log_root'],
            log_dir=dict_args['log_dir'],
            da_task=dict_args['da_task'],
            margin=dict_args['margin'],
            da_lambda=dict_args['da_lambda'],
            learning_rate=dict_args['learning_rate'],
            rampdown_start=dict_args['rampdown_start'],
            rampdown_length=dict_args['rampdown_length'],
            strict=False
        )
    else:
        dict_args['run_id'] = logger.run_id
        dict_args['log_dir'] = os.path.join(dict_args['log_root'], dict_args['run_id'])
        experiment = experiment_class(**dict_args)

    os.makedirs(dict_args['log_dir'], exist_ok=True)

    callbacks = [LearningRateMonitor(logging_interval='epoch'), ModelCheckpoint(dirpath=dict_args['log_dir'], filename='{epoch}', save_last=True)]


    ##
    # Train (optional)
    ##

    # create trainer
    trainer = pl.Trainer(
        max_epochs=dict_args['max_epochs'],
        gpus=dict_args['gpus'],
        logger=logger,
        callbacks=callbacks,
        multiple_trainloader_mode='min_size',
        default_root_dir=dict_args['log_dir'],
        log_every_n_steps=dict_args['log_every_n_steps'],
        flush_logs_every_n_steps=dict_args['log_every_n_steps'] * 2
    )

    # initial validation
    trainer.validate(model=experiment, val_dataloaders=data_module.val_dataloader())
    # train
    trainer.fit(experiment, datamodule=data_module)

    ##
    # Test
    ##

    # test
    trainer.test(experiment, datamodule=data_module)
