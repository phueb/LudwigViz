from itertools import chain
import pandas as pd

from ludwigviz import config
from ludwigviz.app_utils import make_requested


def get_config_values_from_log(logger, config_name, req_completion=True):
    values = set()
    log_entry_dicts = logger.load_log()
    for log_entry_d in log_entry_dicts:
        if req_completion:
            if log_entry_d['timepoint'] == log_entry_d['num_saves']:
                try:
                    config_value = log_entry_d[config_name]
                except KeyError:  # sometimes new config names are added
                    print('rnnlab WARNING: Did not find "{}" in main log.'.format(config_name))
                    continue
                values.add(config_value)
        else:
            values.add(log_entry_d[config_name])
    result = list(values)
    return result


def get_timepoints(logger, model_name):
    last_timepoint = [d['timepoint'] for d in logger.load_log()
                      if d['model_name'] == model_name][0]
    result = list(range(last_timepoint + 1))
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


def get_requested_log_dicts(logger, session, request):
    config_names = make_requested(request, session, 'config_names', default=logger.manipulated_config_names)
    log_dicts = make_log_dicts(logger, config_names)
    log_dict_ids = [int(i) for i in request.args.getlist('log_dict_id')]
    requested_log_dicts = [log_dicts[i] for i in log_dict_ids]
    return requested_log_dicts


def make_log_dicts(logger, config_names):
    log_entry_dicts = logger.load_log()
    # df
    column_names = ['model_name'] + config_names + ['timepoint']
    column_names += ['num_saves'] if 'num_saves' not in config_names else []
    df = pd.DataFrame(data={column_name: [d[column_name] for d in log_entry_dicts]
                            for column_name in column_names})[column_names]
    # make log_dicts
    log_dicts = []
    for config_values, group_df in df.groupby(config_names):
        if not isinstance(config_values, tuple):
            config_values = [config_values]
        model_names = group_df['model_name'].tolist()
        log_dict = {'model_names': model_names,
                    'flavor': model_names[0].split('_')[1],
                    'model_desc': '\n'.join('{}={}'.format(config_name, config_value)
                                            for config_name, config_value in zip(config_names, config_values)),
                    'data_rows': [row.tolist() for ow_id, row in group_df.iterrows()]}
        log_dicts.append(log_dict)
    results = log_dicts[::-1]
    return results


def make_common_timepoint(logger, model_names_list, common_timepoint=config.Interface.common_timepoint):
    timepoints_list_list = []
    for model_names in model_names_list:
        timepoints_list = [logger.get_timepoints(model_name) for model_name in model_names]
        timepoints_list_list.append(timepoints_list)
    sets = [set(list(chain(*l))) for l in timepoints_list_list]
    # default
    if common_timepoint is not None and common_timepoint in set.intersection(*sets):
        print('Using default common timepoint: {}'.format(common_timepoint))
        return common_timepoint
    else:
        print('Did not find default common timepoint.')
    # common
    result = max(set.intersection(*sets))
    print('Found last common timepoint: {}'.format(result))
    return result


def make_log_df(logger, summary_flavors):
    result = pd.read_csv(logger.log_path)
    if not result.empty:
        result['flavor'] = result['model_name'].apply(lambda model_name: model_name.split('_')[1])
        result = result[result['timepoint'] == result['num_saves']]
        result = result[result['flavor'].isin(summary_flavors)]
    return result