import re
import datetime

from ludwigviz import config


regex_digit = re.compile(r'[0-9]+')


def to_param_id(param_name):
    return regex_digit.search(param_name).group()


def get_time_modified(p):
    return datetime.datetime.fromtimestamp(p.lstat().st_mtime).strftime('%H:%M:%S %B %d, %Y')


def get_project_headers_and_rows():

    headers = ['Name', 'Last modified', 'Number of unique runs']

    rows = []
    for p in config.RemoteDirs.research_data.iterdir():
        if not p.name.startswith('.') and p.name not in config.Projects.excluded:
            row = {headers[0]: p.name,
                   headers[1]: get_time_modified(p),
                   headers[2]: len(list(p.glob('runs/param*')))}
            rows.append(row)

    return headers, rows


def make_runs_headers_and_rows(project_name):
    headers = ['Param', 'Time Stamp', 'Replications']
    rows = []
    for p in (config.RemoteDirs.research_data / project_name / 'runs').glob('param*'):
        row = {headers[0]: to_param_id(p.name),
               headers[1]: get_time_modified(p),
               headers[2]: len(list(p.glob('*'))),
               'param_name': p.name,
               }
        rows.append(row)
    return headers, rows
