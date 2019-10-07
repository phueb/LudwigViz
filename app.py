from flask import Flask, redirect, url_for
from flask import render_template
from flask import request, session
import argparse
import json
import socket
import pandas as pd
try:
    import altair as alt
except TypeError:
    raise RuntimeError('altair requires Python > =3.5.3')




import ludwigviz

hostname = socket.gethostname()

app = Flask(__name__)


# ------------------------------------------------ views

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
    if rows:
        header = request.args.get('header') or config.Default.header
        order = request.args.get('order') or config.Default.order
        rows = sort_rows(rows, header, order)

    return render_template('project.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           rows=rows,
                           headers=headers,
                           two_group_btn_names=config.Buttons.two_group_btn_names,
                           any_group_btn_names=config.Buttons.any_group_btn_names)


@app.route('/<string:project_name>/plot', methods=['GET', 'POST'])
def plot(project_name):

    param_names = session['param_names']  # TODO how to plot multiple param_names?

    # TODO is there a way to plot confidence interval?
    # TODO if not, then plot all the individual lines, instead of their average?

    # get all patterns (all possible csv file names) - assume each run has same pattern
    param_p = config.RemoteDirs.research_data / project_name / 'runs' / param_names[0]
    patterns = set([p.name for p in param_p.rglob('*.csv')])

    if not patterns:
        raise LudwigVizNoCsvFound(param_p)
    else:
        print('Detected patterns={}'.format(patterns))

    # iterate over unique df file names (e.g. results_a.csv, results_b.csv)
    json_charts = []
    for pattern in patterns:
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
                y=str(column_name),  # TODO how to plot multiple y?
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

    num_reps = 'not implemented'  # TODO
    param_ids = [to_param_id(param_name) for param_name in param_names]
    return render_template('plots.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           param_names=param_names,
                           param_ids=param_ids,
                           num_reps=num_reps,  # TODO calc this for each param_name separately
                           json_charts=json_charts,
                           )


# ----------------------------------------------- ajax

@app.route('/which_hidden_btns/', methods=['GET'])
def which_hidden_btns():
    """
    count how many checkboxes clicked on project.html to determine which buttons are legal
    """
    num_checkboxes_clicked = int(request.args.get('num_checkboxes_clicked'))
    if num_checkboxes_clicked == 2:
        result = 'both'
    elif num_checkboxes_clicked > 0:
        result = 'any'
    else:
        result = 'none'
    return result


# -------------------------------------------- actions


@app.route('/group_action/<string:project_name>/', methods=['GET', 'POST'])
def group_action(project_name):
    """
    do some operation on one or more runs, e.g. plot results, or delete their data, etc.
    """

    param_names = request.args.getlist('param_name')
    action = request.args.get('action')
    session['param_names'] = param_names

    if action == 'plot':
        return redirect(url_for('plot', project_name=project_name))

    elif action == 'compare':
        raise NotImplementedError

    if action == 'delete_many':
        return redirect(url_for('delete_many',
                                project_name=project_name))

    else:
        raise ValueError('No handler found for action "{}"'.format(request.args.get('action')))

    # TODO add more cases here


@app.route('/delete_many/<string:project_name>', methods=['GET', 'POST'])
def delete_many(project_name):
    if request.args.get('delete_many') is not None:

        print(request.args)

        param_names = request.args.getlist('param_name')

        for param_name in param_names:
            delete_path = config.RemoteDirs.research_data / project_name / 'runs' / param_name
            print('Deleting {}'.format(delete_path))

            # TODO actually implement it

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
    parser.add_argument('--no_debug', action="store_false", default=True, dest='debug',
                        help='Use this for deployment.')
    parser.add_argument('--dummy', action="store_true", default=False, dest='dummy',
                        help='Use a dummy directory - in case mounting does not work')
    namespace = parser.parse_args()

    if namespace.dummy:
        ludwigviz.dummy_data = 'dummy_data'
        print('Using dummy data')

    # import after specifying path to data
    from ludwigviz import config
    from ludwigviz.io import make_runs_headers_and_rows
    from ludwigviz.utils import to_param_id, sort_rows
    from ludwigviz.io import get_project_headers_and_rows

    topbar_dict = {'listing': config.RemoteDirs.research_data,
                   'hostname': hostname,
                   'version': ludwigviz.__version__,
                   'title': ludwigviz.__package__.capitalize()
                   }

    app.secret_key = 'ja0f09'
    app.run(port=5000, debug=namespace.debug, host='0.0.0.0')
