# Built-in libraries
import re
import enum
import os
import collections
import json
import itertools
import glob
from pathlib import Path
import csv
import pandas as pd


def select_slides(slide_path, slide_idx, n_process):
    """
    Function to select a number of slides based on idx (slide_idx) and n_CPUs
    (n_process) in a way that idx should be less than n_slides / n_CPUs.

    Parameters
    ----------
    slide_path : array of str
        array of paths to slides [path/to/slide, ....]
    slide_idx : int
        index of the portion of slides we like to select. In each idx, we select,
        at most n_process slides.
    n_process: int
        number of CPUs for multiprocessing

    Returns
    -------
    slide_path: array of str
        array of paths to the selected slides
    """
    num   = slide_idx
    start = (num-1)*n_process
    end   = num*n_process if num*n_process<len(slide_path) else len(slide_path)
    if start >= len(slide_path):
        raise ValueError(f"Total number of slides are {len(slide_path)}"
                         f" and by using {n_process} CPUs, you are assuming "
                         f"for at least {start+1} slides! --> decrease slide_idx={num}")
    return slide_path[start:end]



def get_slide_names(csv_path):
    df = pd.read_csv(csv_path)
    
    return list(df['slide_id'][:])