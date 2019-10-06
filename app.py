from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
import argparse
import socket

from ludwigviz.io import make_runs_headers_and_rows
from ludwigviz.io import get_project_headers_and_rows

from ludwigviz.app_utils import figs_to_imgs
from ludwigviz.app_utils import LudwigVizEmptySubmission

from ludwigviz import config
from ludwigviz import __version__
from ludwigviz import __package__

hostname = socket.gethostname()

topbar_dict = {'listing': config.RemoteDirs.research_data,
               'hostname': hostname,
               'version': __version__,
               'title': __package__.capitalize()
               }

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    headers, rows = get_project_headers_and_rows()

    return render_template('home.html',
                           topbar_dict=topbar_dict,
                           rows=rows,
                           headers=headers)


@app.route('/<string:project_name>', methods=['GET', 'POST'])
def project(project_name):
    headers, rows = make_runs_headers_and_rows(project_name)

    # sort
    header = request.args.get('header') or config.Default.header
    order = request.args.get('order') or config.Default.order
    rows = sorted(rows, key=lambda d: d[header],
                  reverse=True if order == 'descending' else False)

    return render_template('project.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           rows=rows,
                           headers=headers)


@app.route('/<string:project_name>/<param_name>/', methods=['GET', 'POST'])
def images(project_name, param_name):
    imgs = figs_to_imgs(*figs)
    return render_template('imgs.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           param_name=param_name,
                           num_reps=len(job_names),
                           imgs=imgs)


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


# -------------------------------------------- redirects


@app.route('/log_group_action/', methods=['GET', 'POST'])
def group_action():
    if request.args.get('delete_many') is not None:
        return redirect(url_for('delete_many', log_dict_id=request.args.getlist('log_dict_id')))


@app.route('/delete_all/', methods=['GET', 'POST'])
def delete_all():
    raise NotImplementedError

    return redirect(url_for('project'))


@app.route('/delete_many/', methods=['GET', 'POST'])
def delete_many():
    raise NotImplementedError

    return redirect(url_for('project'))


# -------------------------------------------- error handling

@app.errorhandler(LudwigVizEmptySubmission)  # custom exception
def handle_empty_submission(exception):
    return render_template('error.html',
                           exception=exception,
                           status_code=exception.status_code,
                           topbar_dict=topbar_dict)


@app.errorhandler(500)
def handle_app_error(exception):
    return render_template('error.html',
                           exception=exception,
                           status_code=500,
                           topbar_dict=topbar_dict)


# @app.errorhandler(404)
# def page_not_found(exception):
#     return render_template('error.html',
#                            exception=exception,
#                            status_code=404,
#                            topbar_dict=topbar_dict)


# -------------------------------------------- start app from CL


if __name__ == "__main__":  # pycharm does not use this
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_false", default=True, dest='debug',
                        help='Use this for deployment.')
    argparse_namespace = parser.parse_args()

    app.run(port=5000, debug=argparse_namespace.debug, host='0.0.0.0')
