import os
import glob

import torch
from common.data.load_audio import load_info
import socket

if socket.gethostname() in ['basil', 'chili'] + [f'rechenknecht{i}' for i in range(9)]:
    # set data root for paul's PCs
    BASE_PATH = os.path.join(os.path.expanduser('~'), 'shared', 'DCASE2021', 'task2')
    # TODO: add custom datapath here...
else:
    BASE_PATH = os.path.join(os.path.expanduser('~'), 'shared', 'DCASE2021', 'task2')

MACHINE_TYPES = ['fan', 'gearbox', 'pump', 'slider', 'ToyCar', 'ToyTrain', 'valve']
NUM_SECTIONS = 6
TEST_SECTIONS = [3, 4, 5]


def get_machine_type(path):
    machine_type = os.path.split(os.path.split(os.path.split(path)[0])[0])[1]
    assert machine_type in MACHINE_TYPES
    return machine_type


def get_train(path):
    file_name = os.path.split(path)[1]
    test = file_name.split('_')[3]
    assert test in ['train', 'test']
    return test == 'train'


def get_section(path):
    file_name = os.path.split(path)[1]
    section = int(file_name.split('_')[1])
    assert section in [0, 1, 2, 3, 4, 5]
    return section


def get_source(path):
    file_name = os.path.split(path)[1]
    section = file_name.split('_')[2]
    assert section in ['source', 'target']
    return section == 'source'


def get_anomaly(path):
    file_name = os.path.split(path)[1]
    anomaly = file_name.split('_')[4]
    if anomaly in ['anomaly', 'normal']:
        anomaly = anomaly == 'anomaly'
    else:
        anomaly = -1

    return anomaly


class MachineDataSet(torch.utils.data.Dataset):

    def __init__(
            self,
            data_root=BASE_PATH
    ):
        self.data_root = os.path.join(data_root)
        self.samples = glob.glob(os.path.join(data_root, '**', '**', '**', '*.wav'))
        assert len(self.samples) > 0
        self.samples = [
            {
                'path': s,
                'machine_type': get_machine_type(s),
                'machine_type_idx': MACHINE_TYPES.index(get_machine_type(s)),
                'section': get_section(s),
                'source': get_source(s),
                'train': get_train(s),
                'anomaly': get_anomaly(s),

            } for s in self.samples
        ]

    def __getitem__(self, item):

        sample = self.samples[item].copy()
        info = load_info(sample['path'])
        for k in info:
            sample[k] = info[k]

        return sample

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    md = MachineDataSet()

    md[0]