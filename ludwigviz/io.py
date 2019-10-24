import yaml

from ludwigviz import config
from ludwigviz.utils import to_param_id, get_time_modified, to_param_path


def count_replications(project_name, param_name):
    return len(list(to_param_path(project_name, param_name).glob('*[!.yaml]')))


def get_project_headers_and_rows():
    headers = ['Name', 'Last modified', 'Number of unique jobs']
    rows = []
    for p in config.RemoteDirs.research_data.iterdir():
        if not p.name.startswith('.') and p.name not in config.Projects.excluded:
            num_unique_jobs = len(list(p.glob('runs/param*')))
            if num_unique_jobs == 0:
                continue
            row = {headers[0]: p.name,
                   headers[1]: get_time_modified(p),
                   headers[2]: num_unique_jobs}
            rows.append(row)

    return headers, rows


def make_params_headers_and_rows(project_name):
    headers = ['Param', 'Last modified', 'n']
    rows = []
    for p in (config.RemoteDirs.research_data / project_name / 'runs').glob('param*'):

        # make param2val_reduced
        with (p / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        reduced_param2val = param2val
        del reduced_param2val['job_name']
        del reduced_param2val['param_name']

        tooltip = ''
        for k, v in sorted(reduced_param2val.items()):
            tooltip += f'<p style="margin-bottom: 0px">{k}={v}</p>'

        row = {headers[0]: to_param_id(p.name),
               headers[1]: get_time_modified(p),
               headers[2]: count_replications(project_name, p.name),
               # used, but not displayed in table
               'param_name': p.name,
               'tooltip': tooltip,
               }
        rows.append(row)
    return headers, rows
