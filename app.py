from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
from flask import session
from flask import jsonify
import argparse
from itertools import chain
import pandas as pd

from ludwiglab.app_utils import make_form
from ludwiglab.app_utils import figs_to_imgs
from ludwiglab.app_utils import generate_terms
from ludwiglab.app_utils import get_log_dicts_values
from ludwiglab.log_utils import get_requested_log_dicts, make_log_dicts, make_common_timepoint, make_log_df
from ludwiglab.app_utils import make_model_btn_name_info_dict
from ludwiglab.app_utils import make_requested
from ludwiglab.app_utils import make_template_dict
from ludwiglab.io_utils import write_pca_loadings_files, save_probes_fs_mat, write_probe_neighbors_files, \
    write_acts_tsv_file
from ludwiglab.app_utils import load_configs_dict
from ludwiglab.app_utils import RnnlabAppError
from ludwiglab.app_utils import RnnlabEmptySubmission
from ludwiglab.model_figs import model_btn_name_figs_fn_dict
from ludwiglab.group_figs import group_btn_name_figs_fn_dict
from ludwiglab.model import Model
from ludwiglab import config

from ludwigcluster.logger import Logger
from chjildeshub.hub import Hub


app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
logger = Logger()


@app.route('/', methods=['GET', 'POST'])
def log():
    if not logger.log_path.is_file():
        logger.write_log()
    session.clear()
    config_names = make_requested(request, session, 'config_names', default=logger.manipulated_config_names)
    log_dicts = make_log_dicts(logger, config_names)
    table_headers = ['group_id', 'model_name'] + config_names + ['timepoint', 'num_saves']
    multi_group_btn_names = sorted(AppConfigs.MULTI_GROUP_BTN_NAME_INFO_DICT.keys())
    two_group_btn_names = sorted(AppConfigs.TWO_GROUP_BTN_NAME_INFO_DICT.keys())
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    return render_template('log.html',
                           template_dict=make_template_dict(session),
                           log_dicts=log_dicts,
                           table_headers=table_headers,
                           multi_group_btn_names=multi_group_btn_names,
                           two_group_btn_names=two_group_btn_names,
                           hub_mode=hub_mode,
                           hub_modes=GlobalConfigs.HUB_MODES,
                           all_config_names=logger.all_config_names,
                           hostname=GlobalConfigs.HOSTNAME)


@app.route('/btns/<string:model_name>/', methods=['GET', 'POST'])
def btns(model_name):
    timepoints = logger.get_timepoints(model_name)
    timepoint = int(make_requested(request, session, 'timepoint', default=timepoints[-1]))
    if timepoint > timepoints[-1]:  # in case model doesn't have timepoint matching session timepoint
        timepoint = timepoints[-1]
        session['timepoint'] = [timepoint]
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    model_btn_name_info_dict = make_model_btn_name_info_dict(model_name)
    btn_names = sorted(model_btn_name_info_dict.keys())
    return render_template('btns.html',
                           template_dict=make_template_dict(session),
                           btn_names=btn_names,
                           hub_mode=hub_mode,
                           hub_modes=GlobalConfigs.HUB_MODES,
                           model_name=model_name,
                           timepoint=timepoint,
                           timepoints=timepoints)


@app.route('/btns_action/<string:model_name>/', methods=['GET', 'POST'])
def btns_action(model_name):
    timepoint = int(make_requested(request, session, 'timepoint'))
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    # imgs
    btn_name = request.args.get('btn_name')
    model_btn_name_info_dict = make_model_btn_name_info_dict(model_name)
    imgs_desc, needs_field_input = model_btn_name_info_dict[btn_name]
    if not needs_field_input:
        model = Model(model_name, timepoint)
        model.hub.switch_mode(hub_mode)
        figs = model_btn_name_figs_fn_dict[btn_name](model)
        imgs = figs_to_imgs(*figs)
        return render_template('imgs_model.html',
                               template_dict=make_template_dict(session),
                               model_name=model_name,
                               hub_mode=hub_mode,
                               timepoint=timepoint,
                               imgs=imgs,
                               imgs_desc=imgs_desc)
    else:
        return redirect(url_for('field',
                                model_name=model_name,
                                hub_mode=hub_mode,
                                btn_name=btn_name))


@app.route('/log_group_action/', methods=['GET', 'POST'])
def log_group_action():
    if request.args.get('delete_many') is not None:
        return redirect(url_for('delete_many', log_dict_id=request.args.getlist('log_dict_id')))
    requested_log_dicts = get_requested_log_dicts(logger, session, request)
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    if not requested_log_dicts:
        return 'Please select at least 1 model group.'
    model_names_list = get_log_dicts_values(requested_log_dicts, 'model_names')
    # load models
    common_timepoint = make_common_timepoint(logger, model_names_list)
    if int(common_timepoint) == 0:  # might be due to insertion of P_NOISE or different corpus
        print('///\nrnnlab WARNING: last common mb is 0.\n///')
    model_groups = []
    for model_names in model_names_list:
        group_hub = Hub(mode=hub_mode, **load_configs_dict(model_names[0]))
        models = []
        for n, model_name in enumerate(model_names):
            try:
                timepoint = [timepoint for timepoint in logger.get_timepoints(model_name)
                             if timepoint == common_timepoint][0]
            except IndexError:
                print('Excluded {}. mb_name {} not available'.format(model_name, common_timepoint))
            else:
                model = Model(model_name, timepoint, hub=group_hub)
                models.append(model)
        model_groups.append(models)
    model_descs = get_log_dicts_values(requested_log_dicts, 'model_desc')
    # switch order (color in figure)
    if AppConfigs.SWITCH_MODEL_COMPARISON:  # TODO make REVERSE btn in rnnlab-top to reverse log_dicts
        model_groups = model_groups[::-1]
        model_descs = model_descs[::-1]
    # btn_name
    group_btn_name = request.args.get('group_btn_name')
    try:
        imgs_desc, check_config_names = AppConfigs.MULTI_GROUP_BTN_NAME_INFO_DICT[group_btn_name]
        num_allowed = None
    except KeyError:
        imgs_desc, check_config_names = AppConfigs.TWO_GROUP_BTN_NAME_INFO_DICT[group_btn_name]
        num_allowed = 2
    # checks
    if num_allowed is not None and num_allowed != len(model_groups):
        raise RnnlabAppError('Action allows only {} groups.'.format(
            'more than 0' if num_allowed is None else num_allowed))
    for config_name in check_config_names:
        config_values = [model.configs_dict[config_name] for models in model_groups for model in models]
        if config_values.count(config_values[0]) != len(config_values):
            return 'All models must have same "{}".'.format(config_name)  # TODO raise custom exception
    # imgs
    figs = group_btn_name_figs_fn_dict[group_btn_name](model_groups, model_descs)
    imgs = figs_to_imgs(*figs)
    # save
    if AppConfigs.SAVE_PROBES_BAS_MAT:  # TODO put somewhere else with all other saving fns
        save_probes_fs_mat(model_groups, model_descs)
    return render_template('imgs_group.html',
                           template_dict=make_template_dict(session),
                           imgs_desc=imgs_desc,
                           imgs=imgs)


@app.route('/log_global_action/', methods=['GET', 'POST'])
def log_global_action():
    if request.args.get('delete_all') is not None:
        return redirect(url_for('delete_all'))
    elif request.args.get('summarize_all') is not None:
        return redirect(url_for('get_stats'))
    else:
        return 'rnnlab: Invalid request "{}".'.format(request.args)  # TODO put all such error messages into a formatted html


@app.route('/autocomplete/', methods=['GET'])
def autocomplete():
    return jsonify(json_list=session['autocomplete_list'])  # TODO cookie too large


@app.route('/which_hidden_btns/', methods=['GET'])
def which_hidden_btns():
    num_checkboxes_clicked = int(request.args.get('num_checkboxes_clicked'))
    if num_checkboxes_clicked == 2:
        result = 'both'
    elif num_checkboxes_clicked > 0:
        result = 'any'
    else:
        result = 'none'
    return result


@app.route('/field/<string:model_name>/<string:btn_name>', methods=['GET', 'POST'])
def field(model_name, btn_name):
    timepoint = int(make_requested(request, session, 'timepoint'))
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    model = Model(model_name, timepoint)
    model.hub.switch_mode(hub_mode)
    # form
    btn_name_info_dict = make_model_btn_name_info_dict(model_name)
    imgs_desc, valid_type = btn_name_info_dict[btn_name]
    form = make_form(model, request, AppConfigs.DEFAULT_FIELD_STR, valid_type)
    # autocomplete
    if valid_type == 'probe':
        session['autocomplete_list'] = list(model.hub.probe_store.types)
    elif valid_type == 'cat':
        session['autocomplete_list'] = list(model.hub.probe_store.cats)
    elif valid_type == 'term':
        session['autocomplete_list'] = list(model.hub.train_terms.types)
    else:
        session['autocomplete_list'] = []
    # request
    if form.validate():
        field_input = form.field.data.split()
        figs = model_btn_name_figs_fn_dict[btn_name](model, field_input)
        imgs = figs_to_imgs(*figs)
        return render_template('imgs_model.html',
                               template_dict=make_template_dict(session),
                               model_name=model_name,
                               timepoint=timepoint,
                               hub_mode=hub_mode,
                               imgs=imgs,
                               imgs_desc=imgs_desc)
    else:
        return render_template('field.html',
                               template_dict=make_template_dict(session),
                               model_name=model_name,
                               timepoint=timepoint,
                               hub_mode=hub_mode,
                               form=form,
                               btn_name=btn_name)


@app.route('/delete_all/', methods=['GET', 'POST'])
def delete_all():
    config_names = session['config_names']
    log_dicts = make_log_dicts(logger, config_names)
    model_names_list = get_log_dicts_values(log_dicts, 'model_names')
    for model_name in chain(*model_names_list):
        logger.delete_model(model_name)
    return redirect(url_for('log'))


@app.route('/delete_many/', methods=['GET', 'POST'])
def delete_many():
    requested_log_dicts = get_requested_log_dicts(logger, session, request)
    model_names_list = get_log_dicts_values(requested_log_dicts, 'model_names')
    for model_name in chain(*model_names_list):
        logger.delete_model(model_name)
    return redirect(url_for('log'))


@app.route('/get_stats/', methods=['GET', 'POST'])
def get_stats():
    config_names = session['config_names']
    # summary_flavors
    flavors_in_logs = logger.get_config_values_from_log('flavor')
    summary_flavors = make_requested(request, session, 'summary_flavors', default=flavors_in_logs)
    log_df = make_log_df(logger, summary_flavors)
    # stats_table
    if not log_df.empty:
        group_tables = []
        for name, g in log_df.groupby(config_names):
            group_table = g[['model_name'] + config_names].to_html(
                classes='mdl-data-table', index_names=False, index=False).replace('border="1"', 'border="0"')
            group_tables.append(group_table)
        imgs_desc = 'Summary Statistics'
    else:
        group_tables = []
        imgs_desc = 'No logs matching criteria found.'
    return render_template('stats.html',
                           template_dict=make_template_dict(session),
                           imgs_desc=imgs_desc,
                           group_tables=group_tables,
                           summary_flavors=summary_flavors,
                           flavors_in_logs=flavors_in_logs,
                           config_names=config_names)


@app.route('/show_windows/<string:model_name>/', methods=['GET', 'POST'])
def show_windows(model_name):
    timepoint = 0
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    model = Model(model_name, timepoint)
    model.hub.switch_mode(hub_mode)
    windows_tables = []
    # form
    form = make_form(model, request, AppConfigs.DEFAULT_FIELD_STR, 'probe')
    # make windows_tables
    if form.validate():
        probes = form.field.data
        for probe in probes.split():
            probe_id = model.hub.probe_store.probe_id_dict[probe]
            probe_x_mat = model.hub.probe_x_mats[model.hub.probe_store.probe_id_dict[probe]]
            probe_context_df = pd.DataFrame(index=[probe_id] * len(probe_x_mat), data=probe_x_mat)
            table_df = probe_context_df.apply(
                lambda term_ids: [model.hub.train_terms.types[term_id] for term_id in term_ids])
            windows_table = table_df.to_html(index=False, header=False).replace('border="1"', 'border="0"')
            windows_tables.append(windows_table)
    return render_template('windows.html',
                           template_dict=make_template_dict(session),
                           model_name=model_name,
                           timepoint=model.timepoint,
                           form=form,
                           windows_tables=windows_tables)


@app.route('/generate/<string:model_name>', methods=['GET', 'POST'])  # TODO use celery queue to compute on gpu node, rather than server's cpu?
def generate(model_name):
    phrase = None
    timepoint = int(make_requested(request, session, 'timepoint'))
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    model = Model(model_name, timepoint)
    model.hub.switch_mode(hub_mode)
    # task_id
    task_btn_str = make_requested(request, session, 'task_btn_str', default='Predict next terms')
    task_names = ['predict'] + model.task_names
    task_name = AppConfigs.TASK_BTN_STR_TASK_NAME_DICT[task_btn_str]
    task_id = task_names.index(task_name)
    # task_btn_strs
    task_btn_strs = [task_btn_str for task_btn_str in sorted(AppConfigs.TASK_BTN_STR_TASK_NAME_DICT.keys())
                     if AppConfigs.TASK_BTN_STR_TASK_NAME_DICT[task_btn_str] in task_names]
    # form
    input_str = 'Type phrase here'
    form = make_form(model, request, input_str, 'term')
    # make output_dict
    output_dict = {}
    if form.validate():
        phrase = form.field.data
        terms = phrase.split()
        for num_samples in AppConfigs.NUM_SAMPLES_LIST:
            output_dict[num_samples] = generate_terms(model, terms, task_id, num_samples=num_samples)
    return render_template('generate.html',
                           template_dict=make_template_dict(session),
                           model_name=model_name,
                           timepoint=model.timepoint,
                           form=form,
                           phrase=phrase,
                           output_dict=output_dict,
                           num_samples_list=AppConfigs.NUM_SAMPLES_LIST,
                           task_btn_strs=task_btn_strs)


@app.route('/delete_one/<string:model_name>/', methods=['GET', 'POST'])
def delete_one(model_name):
    logger.delete_model(model_name)
    return redirect(url_for('log'))


@app.route('/export_acts/<string:model_name>/', methods=['GET', 'POST'])
def export_acts(model_name):
    timepoint = int(make_requested(request, session, 'timepoint'))
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    model = Model(model_name, timepoint)
    model.hub.switch_mode(hub_mode)
    write_acts_tsv_file(model)
    return redirect(url_for('btns', model_name=model_name))


@app.route('/export_neighbors/<string:model_name>/', methods=['GET', 'POST'])
def export_neighbors(model_name):
    timepoint = int(make_requested(request, session, 'timepoint'))
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    model = Model(model_name, timepoint)
    model.hub.switch_mode(hub_mode)
    write_probe_neighbors_files(model)
    return redirect(url_for('btns', model_name=model_name))


@app.route('/export_pca_loadings/<string:model_name>/', methods=['GET', 'POST'])
def export_pca_loadings(model_name):
    timepoint = int(make_requested(request, session, 'timepoint'))
    hub_mode = make_requested(request, session, 'hub_mode', default=GlobalConfigs.HUB_MODES[0])
    model = Model(model_name, timepoint)
    model.hub.switch_mode(hub_mode)
    write_pca_loadings_files(model, is_terms=True, freq_thr=0)
    write_pca_loadings_files(model, is_terms=True, freq_thr=100)
    write_pca_loadings_files(model, is_terms=False, freq_thr=0)
    return redirect(url_for('btns', model_name=model_name))


@app.errorhandler(RnnlabEmptySubmission)  # custom exception
def handle_empty_submission(error):
    return render_template('error.html',
                           exception=error,
                           status_code=error.status_code,
                           template_dict=make_template_dict(session))


@app.errorhandler(RnnlabAppError)  # custom exception
def handle_empty_submission(error):
    return render_template('error.html',
                           exception=error,
                           status_code=error.status_code,
                           template_dict=make_template_dict(session))


@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html',
                           exception=error,
                           status_code=404,
                           template_dict=make_template_dict(session))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodebug', action="store_false", default=True, dest='debug',
                        help='Use this for deployment.')
    return parser


if __name__ == "__main__":
    ap = arg_parser()
    argparse_namespace = ap.parse_args()
    app.run(port=5000, debug=argparse_namespace.debug, host='0.0.0.0')
