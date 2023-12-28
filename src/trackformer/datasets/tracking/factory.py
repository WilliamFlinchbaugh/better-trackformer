# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Factory of tracking datasets.
"""
from typing import Union

from torch.utils.data import ConcatDataset
import os

from .demo_sequence import DemoSequence
from .mot_wrapper import MOT17Wrapper, MOT20Wrapper, MOTS20Wrapper, BASEWrapper, DistanceWrapper, MovementWrapper, LightingWrapper

DATASETS = {}

# Fill all available datasets, change here to modify / add new datasets.
        
for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for dets in ['DPM', 'FRCNN', 'SDP', 'ALL']:
        name = f'MOT17-{split}'
        if dets:
            name = f"{name}-{dets}"
        DATASETS[name] = (
            lambda kwargs, split=split, dets=dets: MOT17Wrapper(split, dets, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '03', '04', '05',
              '06', '07', '08']:
    name = f'MOT20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOT20Wrapper(split, **kwargs))


for split in ['TRAIN', 'TEST', 'ALL', '01', '02', '05', '06', '07', '09', '11', '12']:
    name = f'MOTS20-{split}'
    DATASETS[name] = (
        lambda kwargs, split=split: MOTS20Wrapper(split, **kwargs))

DATASETS['DEMO'] = (lambda kwargs: [DemoSequence(**kwargs), ])

# Custom Datasets

for i in range(4):
    for split in ['TRAIN', 'TEST', 'ALL']:
        name = f'BASE-{split}-{i}'
        DATASETS[name] = (lambda kwargs, split=split, name=name: BASEWrapper(split, name, 'base', **kwargs))
        
for split in ['pendleton_1', 'pendleton_2', 'pendleton_3', 'pendleton_4', 'pendleton_5', 'pendleton_6', 'pendleton_7', 'pendleton_8', 'pendleton_9', 'pendleton_10', 'yemen_1', 'yemen_2']:
    DATASETS[split] = (lambda kwargs, split=split, name=split: BASEWrapper(split, name, 'base', **kwargs))

for distance in ['25m', '50m', '100m', '150m', '200m', '300m', '400m', '500m', '750m', '1000m']:
    DATASETS[distance] = (lambda kwargs, split=distance: DistanceWrapper(split, **kwargs))
    
for seq in ['pendleton_1', 'pendleton_2', 'pendleton_3', 'pendleton_4', 'pendleton_5', 'pendleton_6', 'pendleton_7', 'pendleton_10', 'yemen_1', 'yemen_2']:
    for distance in ['25m', '50m', '100m', '150m', '200m', '300m', '400m', '500m', '750m', '1000m']:
        DATASETS[f'{seq}_{distance}'] = (lambda kwargs, split=f'{distance}/{seq}_{distance}': DistanceWrapper(split, **kwargs))
    
for speed in ['0ms', '5ms', '10ms', '20ms', '40ms']:
    DATASETS[speed] = (lambda kwargs, split=speed: MovementWrapper(split, **kwargs))
    
for seq in ['pendleton_1', 'pendleton_2', 'pendleton_3', 'pendleton_4', 'pendleton_5', 'pendleton_6', 'pendleton_7', 'pendleton_10', 'yemen_1', 'yemen_2']:
    for speed in ['0ms', '5ms', '10ms', '20ms', '40ms']:
        DATASETS[f'{seq}_{speed}'] = (lambda kwargs, split=f'{speed}/{seq}_{speed}': MovementWrapper(split, **kwargs))
    
for time in ['1s', '2s', '5s', '10s', '20s']:
    DATASETS[time] = (lambda kwargs, split=time: LightingWrapper(split, **kwargs))

class TrackDatasetFactory:
    """A central class to manage the individual dataset loaders.

    This class contains the datasets. Once initialized the individual parts (e.g. sequences)
    can be accessed.
    """

    def __init__(self, datasets: Union[str, list], **kwargs) -> None:
        """Initialize the corresponding dataloader.

        Keyword arguments:
        datasets --  the name of the dataset or list of dataset names
        kwargs -- arguments used to call the datasets
        """
        if isinstance(datasets, str):
            datasets = [datasets]

        self._data = None
        for dataset in datasets:
            assert dataset in DATASETS, f"[!] Dataset not found: {dataset}"

            if self._data is None:
                self._data = DATASETS[dataset](kwargs)
            else:
                self._data = ConcatDataset([self._data, DATASETS[dataset](kwargs)])

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
