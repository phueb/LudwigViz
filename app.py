from flask import Flask, redirect, url_for
from flask import render_template
from flask import request
import argparse
import json
import socket
import pandas as pd
try:
    import altair as alt
except TypeError:
    raise RuntimeError('altair requires Python > =3.5.3')


from ludwigviz.io import make_runs_headers_and_rows
from ludwigviz.utils import to_param_id, sort_rows
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
    rows = sort_rows(rows, header, order)

    return render_template('project.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           rows=rows,
                           headers=headers)


@app.route('/<string:project_name>/<param_name>/', methods=['GET', 'POST'])
def images(project_name, param_name):

    # TODO is there a way to plot confidence interval?
    # TODO if not, then plot all the individual lines, instead of their average?

    # get all file_names matching *.csv
    param_p = config.RemoteDirs.research_data / project_name / 'runs' / param_name
    df_file_names = [p.name for p in param_p.rglob('*.csv')]

    if not df_file_names:
        raise LudwigVizNoCsvFound(param_p)

    # iterate over unique df file names
    json_charts = []
    for pattern in set(df_file_names):
        # get all dfs matching pattern
        dfs = [pd.read_csv(p, index_col=0) for p in param_p.rglob(pattern)]

        # average columns with the same name
        concatenated_df = pd.concat(dfs, axis=1)
        df = concatenated_df.groupby(by=concatenated_df.columns, axis=1).mean()
        df['x'] = df.index

        # iterate over column names, making a chart for each
        for column_name in df.columns:
            if column_name == 'x':
                continue
            # make interactive chart and convert to json object
            chart = alt.Chart(df).mark_line().encode(
                x='x',
                y=str(column_name),
            ).interactive()
            # to json
            json_str = chart.to_json()
            json_chart = json.loads(json_str)
            # set title and size
            json_chart['config']['view']['height'] *= config.Chart.scale_factor
            json_chart['config']['view']['width'] *= config.Chart.scale_factor
            json_chart['title'] = pattern.rstrip('.csv').capitalize()
            # collect chart
            json_charts.append(json_chart)

    num_reps = len(df_file_names) // len(set(df_file_names))
    return render_template('imgs.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           param_name=param_name,
                           param_id=to_param_id(param_name),
                           num_reps=num_reps,
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

class LudwigVizNoCsvFound(Exception):
    def __init__(self, param_p, status_code=500):
        Exception.__init__(self)
        self.message = 'LudwigViz: Did not find any csv files in {}'.format(param_p)
        if status_code is not None:
            self.status_code = status_code


@app.errorhandler(LudwigVizNoCsvFound)
def handle_not_found_error(exception):
    return render_template('error.html',
                           message=exception.message,
                           status_code=500,
                           topbar_dict=topbar_dict)


@app.errorhandler(500)
def handle_app_error(exception):
    return render_template('error.html',
                           message=exception,
                           status_code=500,
                           topbar_dict=topbar_dict)


@app.errorhandler(404)
def page_not_found(exception):
    return render_template('error.html',
                           message=exception,
                           status_code=404,
                           topbar_dict=topbar_dict)


# -------------------------------------------- start app from CL


if __name__ == "__main__":  # pycharm does not use this
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_false", default=True, dest='debug',
                        help='Use this for deployment.')
    argparse_namespace = parser.parse_args()

    app.run(port=5000, debug=argparse_namespace.debug, host='0.0.0.0')
