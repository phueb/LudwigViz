from pathlib import Path
import os

from ludwigviz import dummy_data, mnt_point


class RemoteDirs:

    if dummy_data is None:
        research_data = Path(mnt_point) / 'research_data'
        if not os.path.ismount(str(research_data)):
            print('WARNING: {} not mounted.'
                  'Using dummy directory for development'.format(research_data))
            research_data = Path(mnt_point) / 'dummy_data'
    # use dummy
    else:
        research_data = Path(dummy_data)


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'ludwigviz'
    static = root / 'static'
    templates = root / 'templates'


class Default:
    header = 'Param'
    order = 'ascending'


class Buttons:
    any_group_btn_names = ['plot']  # TODO use
    two_group_btn_names = ['compare']


class Time:
    format = '%H:%M:%S %B %d, %Y'


class Projects:
    excluded = ['stdout', 'Example']


class Chart:
    x_name = 'x'
    scale_factor = 1.4