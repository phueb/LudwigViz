from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
import argparse
import json
import socket
import altair as alt
import pandas as pd

from ludwigviz.io import make_runs_headers_and_rows
from ludwigviz.io import to_param_id
from ludwigviz.io import get_project_headers_and_rows


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

    # TODO there are multiple different csv files in each job_dir
    # TODO each df can have multiple different columns
    pattern = 'results_a.csv'  # TODO how to load only csv files with same name?

    # concatenate all results
    param_p = config.RemoteDirs.research_data / project_name / 'runs' / param_name
    dfs = [pd.read_csv(p, index_col=0) for p in param_p.rglob(pattern)]

    # average columns with the same name
    concatenated_df = pd.concat(dfs, axis=1)
    df = concatenated_df.groupby(by=concatenated_df.columns, axis=1).mean()
    df['x'] = df.index

    # iterate over column names, making a chart for each
    json_charts = []
    for column_name in df.columns:
        if column_name == 'x':
            continue
        # make interactive chart and convert to json object
        chart = alt.Chart(df).mark_line().encode(
            x='x',
            y=str(column_name),
        ).interactive()
        json_str = chart.to_json()
        json_chart = json.loads(json_str)
        json_chart['config']['view']['height'] *= config.Chart.scale_factor
        json_chart['config']['view']['width'] *= config.Chart.scale_factor
        # collect chart
        json_charts.append(json_chart)

    return render_template('imgs.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           param_name=param_name,
                           param_id=to_param_id(param_name),
                           num_reps=len(dfs),
                           json_charts=json_charts,
                           )


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

@app.errorhandler(500)
def handle_app_error(exception):
    return render_template('error.html',
                           exception=exception,
                           status_code=500,
                           topbar_dict=topbar_dict)


@app.errorhandler(404)
def page_not_found(exception):
    return render_template('error.html',
                           exception=exception,
                           status_code=404,
                           topbar_dict=topbar_dict)


# -------------------------------------------- start app from CL


if __name__ == "__main__":  # pycharm does not use this
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_false", default=True, dest='debug',
                        help='Use this for deployment.')
    argparse_namespace = parser.parse_args()

    app.run(port=5000, debug=argparse_namespace.debug, host='0.0.0.0')
