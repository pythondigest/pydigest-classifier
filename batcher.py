import json
from itertools import islice, chain
import config

"""
Contains various batchers for memory-efficient NN training
"""


def batch(iterable, size): # general batcher fo arbi
    """General batcher for arbitrary iterable and chunk size"""
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([next(batchiter)], batchiter)

