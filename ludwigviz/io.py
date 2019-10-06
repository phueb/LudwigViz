from ludwigviz import config
from ludwigviz.utils import to_param_id, get_time_modified


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


def make_runs_headers_and_rows(project_name):
    headers = ['Param', 'Last modified', 'Replications']
    rows = []
    for p in (config.RemoteDirs.research_data / project_name / 'runs').glob('param*'):
        row = {headers[0]: to_param_id(p.name),
               headers[1]: get_time_modified(p),
               headers[2]: len(list(p.glob('*'))),
               'param_name': p.name,
               }
        rows.append(row)
    return headers, rows
