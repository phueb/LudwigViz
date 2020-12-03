from pathlib import Path
import os


class Dirs:

    research_data = Path('/media') / 'research_data'
    if not os.path.ismount(str(research_data)):
        print('WARNING: {} not mounted.'
              'Using dummy directory for development'.format(research_data))
        research_data = Path('dummy_data')

    root = Path(__file__).parent.parent
    src = root / 'ludwigviz'
    static = root / 'static'
    templates = root / 'templates'


class Default:
    header = 'Param'
    order = 'ascending'


class Buttons:
    any_group_btn_names = ['plot']
    two_group_btn_names = []


class Time:
    format = '%H:%M %B %d'


class Projects:
    excluded = ['stdout', 'Example']


class Chart:
    x_name = 'step'  # this label may not be correct for all users
    scale_factor = 1.4

    # todo make limits specific to a project, not to all projects
    name2y_lims = {
        'devel_pps': [1.0, 200.0],
    }
