# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT wrapper which combines sequences to a dataset.
"""
from torch.utils.data import Dataset
import json
import os

from .mot17_sequence import MOT17Sequence
from .mot20_sequence import MOT20Sequence
from .mots20_sequence import MOTS20Sequence
from .base_sequence import BASESequence
from .dist_sequence import DistanceSequence
from .movement_sequence import MovementSequence
from .lighting_sequence import LightingSequence

class MOT17Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, dets: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT17Sequence dataset
        """
        train_sequences = [
            'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
            'MOT17-10', 'MOT17-11', 'MOT17-13']
        test_sequences = [
            'MOT17-01', 'MOT17-03', 'MOT17-06', 'MOT17-07',
            'MOT17-08', 'MOT17-12', 'MOT17-14']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT17-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT17-{split}"]
        else:
            raise NotImplementedError("MOT17 split not available.")

        self._data = []
        for seq in sequences:
            if dets == 'ALL':
                self._data.append(MOT17Sequence(seq_name=seq, dets='DPM', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='FRCNN', **kwargs))
                self._data.append(MOT17Sequence(seq_name=seq, dets='SDP', **kwargs))
            else:
                self._data.append(MOT17Sequence(seq_name=seq, dets=dets, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

class BASEWrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, name:str, root: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        print(name)
        cwd = os.getcwd()
        if (split != name):
            base_split = name[-1]
            with open(os.path.join(cwd, 'data', root, 'splits.json'), 'r') as f:
                splits = json.load(f)
                this_split = splits[base_split]
                train_sequences = this_split['train']
                test_sequences = this_split['test']
                
        else: 
            train_sequences = ['pendleton_1', 'pendleton_2', 'pendleton_3', 'pendleton_4', 'pendleton_5', 'pendleton_6', 'pendleton_7', 'pendleton_8', 'pendleton_9', 'pendleton_10', 'yemen_1', 'yemen_2']
            test_sequences = ['pendleton_1', 'pendleton_2', 'pendleton_3', 'pendleton_4', 'pendleton_5', 'pendleton_6', 'pendleton_7', 'pendleton_8', 'pendleton_9', 'pendleton_10', 'yemen_1', 'yemen_2']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif split in train_sequences + test_sequences:
            sequences = [split]
        else:
            raise NotImplementedError("Base split not available.")
        
        self._data = []
        for seq in sequences:
            self._data.append(BASESequence(seq_name=seq, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
        
class DistanceWrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """  
        
        self._data = []
        
        single_seq = True if (split.find('pendleton') != -1 or split.find('yemen') != -1) else False
        if single_seq:
            self._data.append(DistanceSequence(seq_name=split.split('/')[-1], split_folder=split.split('/')[0], dets=None, **kwargs))
            
        else:  
            sequences = os.listdir(os.path.join('data', 'distances', split))
            sequences = sorted(sequences)
            for seq in sequences:
                self._data.append(DistanceSequence(seq_name=seq, split_folder=split, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
    
class MovementWrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        
        self._data = []
        
        single_seq = True if (split.find('pendleton') != -1 or split.find('yemen') != -1) else False
        if single_seq:
            self._data.append(MovementSequence(seq_name=split.split('/')[-1], split_folder=split.split('/')[0], dets=None, **kwargs))
            
        else:  
            sequences = os.listdir(os.path.join('data', 'movement', split))
            sequences = sorted(sequences)
            for seq in sequences:
                self._data.append(MovementSequence(seq_name=seq, split_folder=split, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

class LightingWrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        
        cwd = os.getcwd()
        sequences = os.listdir(os.path.join('data', 'lighting', split))
        sequences = sorted(sequences)

        self._data = []
        for seq in sequences:
            self._data.append(LightingSequence(seq_name=seq, split_folder=split, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]

class MOT20Wrapper(Dataset):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOT20Sequence dataset
        """
        train_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05',]
        test_sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08',]

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOT20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT20-{split}"]
        else:
            raise NotImplementedError("MOT20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOT20Sequence(seq_name=seq, dets=None, **kwargs))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        return self._data[idx]
        
        
class MOTS20Wrapper(MOT17Wrapper):
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, split: str, **kwargs) -> None:
        """Initliazes all subset of the dataset.

        Keyword arguments:
        split -- the split of the dataset to use
        kwargs -- kwargs for the MOTS20Sequence dataset
        """
        train_sequences = ['MOTS20-02', 'MOTS20-05', 'MOTS20-09', 'MOTS20-11']
        test_sequences = ['MOTS20-01', 'MOTS20-06', 'MOTS20-07', 'MOTS20-12']

        if split == "TRAIN":
            sequences = train_sequences
        elif split == "TEST":
            sequences = test_sequences
        elif split == "ALL":
            sequences = train_sequences + test_sequences
            sequences = sorted(sequences)
        elif f"MOTS20-{split}" in train_sequences + test_sequences:
            sequences = [f"MOTS20-{split}"]
        else:
            raise NotImplementedError("MOTS20 split not available.")

        self._data = []
        for seq in sequences:
            self._data.append(MOTS20Sequence(seq_name=seq, **kwargs))
