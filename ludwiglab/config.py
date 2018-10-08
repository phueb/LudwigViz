from pathlib import Path


class Dirs:
    lab = Path('/') / 'media' / 'lab'
    src = Path().cwd() / 'ludwigcluster'


class Interface:
    common_timepoint = 10


class Figs:
    NUM_PCS = 5