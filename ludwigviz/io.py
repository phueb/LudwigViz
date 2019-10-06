import re
import datetime

from ludwigviz import config


regex_workers = re.compile(r'(hoff|hebb)')  # TODO use SFTP.worker_names
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


def make_param_rows(project_name):

    headers = ['Worker', 'Time Stamp', 'Replications']
    rows = []
    for p in (config.RemoteDirs.research_data / project_name / 'runs').glob('param*'):

        # TODO implement
        print(regex_workers.search(p.name))

        row = {'param_name': p.name,
               headers[0]: regex_workers.search(p.name).group(),
               headers[1]: get_time_modified(p),
               headers[2]: p.name,
               }
        rows.append(row)
    return headers, rows


def get_config_values_from_log(logger, config_name, req_completion=True):
    values = set()
    log_entry_dicts = logger.load_log()
    for log_entry_d in log_entry_dicts:
        if req_completion:
            if log_entry_d['timepoint'] == log_entry_d['num_saves']:
                try:
                    config_value = log_entry_d[config_name]
                except KeyError:  # sometimes new config names are added
                    print('LudwigViz WARNING: Did not find "{}" in main log.'.format(config_name))
                    continue
                values.add(config_value)
        else:
            values.add(log_entry_d[config_name])
    result = list(values)
    return result


def get_manipulated_config_names(logger):
    """
    Returns list of all config_names for which there is more than one unique value in all logs
    """
    result = []
    for config_name in logger.all_config_names:
        config_values = get_config_values_from_log(logger, config_name, req_completion=False)
        is_manipulated = True if len(list(set(config_values))) > 1 else False
        if is_manipulated and config_name in logger.all_config_names:
            result.append(config_name)
    # if empty
    if not result:
        result = ['flavor']
    return result


