from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
from flask import session
from flask import jsonify
import argparse
from itertools import chain
import pandas as pd

from childeshub.hub import Hub

from ludwigviz.log_utils import get_requested_log_dicts
from ludwigviz.log_utils import make_log_dicts

from ludwigviz.app_utils import make_form
from ludwigviz.app_utils import figs_to_imgs
from ludwigviz.app_utils import get_log_dicts_values
from ludwigviz.app_utils import make_topbar_dict
from ludwigviz.app_utils import LudwigVizEmptySubmission

from ludwigviz import config


app = Flask(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'


@app.route('/', methods=['GET', 'POST'])
def home():

    project_names = list(config.RemoteDirs.research_data.glob('*'))

    table_headers = ['header1', 'header2']

    return render_template('home.html',
                           topbar_dict=make_topbar_dict(session),
                           project_names=project_names,
                           table_headers=table_headers)


@app.route('/<string:project_name>/', methods=['GET', 'POST'])
def project(project_name):
    log_dicts = make_log_dicts(logger, config_names)
    table_headers = ['group_id', 'model_name'] + config_names + ['timepoint', 'num_saves']
    return render_template('project.html',
                           topbar_dict=make_topbar_dict(session),
                           log_dicts=log_dicts,
                           table_headers=table_headers,
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
                           topbar_dict=make_topbar_dict(session),
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
        return render_template('imgs.html',
                               topbar_dict=make_topbar_dict(session),
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
    # form
    form = make_form(model, request, config.Default.field_input, config.Default.valid_type)
    # autocomplete
    if valid_type == 'probe':
        session['autocomplete_list'] = list(hub.probe_store.types)
    elif valid_type == 'cat':
        session['autocomplete_list'] = list(hub.probe_store.cats)
    elif valid_type == 'term':
        session['autocomplete_list'] = list(hub.train_terms.types)
    else:
        session['autocomplete_list'] = []
    # request
    if form.validate():
        field_input = form.field.data.split()
        figs = model_btn_name_figs_fn_dict[btn_name](model, field_input)
        imgs = figs_to_imgs(*figs)
        return render_template('imgs.html',
                               template_dict=make_topbar_dict(session),
                               model_name=model_name,
                               timepoint=timepoint,
                               hub_mode=hub_mode,
                               imgs=imgs,
                               imgs_desc=imgs_desc)
    else:
        return render_template('field.html',
                               template_dict=make_topbar_dict(session),
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
    return redirect(url_for('project'))


@app.route('/delete_many/', methods=['GET', 'POST'])
def delete_many():
    requested_log_dicts = get_requested_log_dicts(logger, session, request)
    model_names_list = get_log_dicts_values(requested_log_dicts, 'model_names')
    for model_name in chain(*model_names_list):
        logger.delete_model(model_name)
    return redirect(url_for('project'))


@app.errorhandler(LudwigVizEmptySubmission)  # custom exception
def handle_empty_submission(exception):
    return render_template('error.html',
                           exception=exception,
                           status_code=exception.status_code,
                           template_dict=make_topbar_dict(session))


@app.errorhandler(500)
def handle_app_error(exception):
    return render_template('error.html',
                           exception=exception,
                           status_code=500,
                           template_dict=make_topbar_dict(session))


@app.errorhandler(404)
def page_not_found(exception):
    return render_template('error.html',
                           exception=exception,
                           status_code=404,
                           template_dict=make_topbar_dict(session))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_false", default=True, dest='debug',
                        help='Use this for deployment.')
    return parser


if __name__ == "__main__":
    ap = arg_parser()
    argparse_namespace = ap.parse_args()
    app.run(port=5000, debug=argparse_namespace.debug, host='0.0.0.0')
