
"""
BASE sequence dataset.
"""

from .mot17_sequence import MOT17Sequence


class MovementSequence(MOT17Sequence):
    """Multiple Object Tracking (MOT20) Dataset.

    This dataloader is designed so that it can handle only one sequence,
    if more have to be handled one should inherit from this class.
    """
    data_folder = 'movement'
