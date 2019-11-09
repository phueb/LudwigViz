from flask import Flask, redirect, url_for
from flask import render_template
from flask import request, session
import argparse
import socket
import yaml

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
    headers, rows = make_params_headers_and_rows(project_name)

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
    """
    is requested in two ways:
    1. directly clicking on row in project.html
    2. by selecting multiple runs and clicking on "plot" button

    in case 1, param_names is retrieved from request object.
    in case 2, param_names is retrieved from session object (because of redirect)

    check request object first, and only then check session
    """

    param_names = request.args.getlist('param_name') or session.get('param_names')

    # TODO is there a way to plot confidence interval?
    # TODO if not, then plot all the individual lines, instead of their average?

    # get all patterns (all possible csv file names) - assume each run has same pattern
    first_param_path = to_param_path(project_name, param_names[0])
    patterns = set([p.name for p in first_param_path.rglob('*.csv')])

    if not patterns:
        raise LudwigVizNoCsvFound(first_param_path)
    else:
        print('Detected patterns={}'.format(patterns))

    # iterate over unique df file names (e.g. results_a.csv, results_b.csv)
    json_charts = []
    for pattern in patterns:
        print('pattern="{}"'.format(pattern))
        # get data frame where each column represents a mean for a particular param_name
        data = aggregate_data(project_name, param_names, pattern)
        # make chart
        title = pattern.rstrip('.csv').capitalize()
        column_name = data.columns[0]
        json_chart = make_json_chart(data, column_name, title)
        # collect chart
        json_charts.append(json_chart)

    # get number of reps for each param_name
    param_name2n = {param_name: count_replications(project_name, param_name)
                    for param_name in param_names}
    return render_template('plots.html',
                           topbar_dict=topbar_dict,
                           project_name=project_name,
                           param_names=param_names,
                           param_name2n=param_name2n,
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


@app.route('/compare_params/<string:project_name>/', methods=['GET', 'POST'])
def compare_params(project_name):
    excluded_keys = ['param_name', 'job_name', 'save_path', 'project_path']

    if request.method == 'POST':
        param_names = request.get_json()['param_names']
        print('param_names:')
        print(param_names)
    else:
        param_names = []

    param2val_list = []
    for param_name in param_names:
        param_path = to_param_path(project_name, param_name)
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        param2val_list.append(param2val)

    message = ''
    keys = [k for k in param2val_list[0].keys() if k not in excluded_keys]
    for key in keys:
        param_values = [param2val[key] for param2val in param2val_list]
        if len(set(param_values)) != 1:  # param_values differ between configurations
            message += '<p><b>{}</b>={}</p>'.format(key, param_values)
    return message + '<p>(shown in order of configurations selected)</p>'


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

    if action == 'delete_many':
        return redirect(url_for('delete_many',
                                project_name=project_name))

    else:
        raise ValueError('No handler found for action "{}"'.format(request.args.get('action')))


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
    from ludwigviz.io import make_params_headers_and_rows
    from ludwigviz.utils import sort_rows
    from ludwigviz.utils import to_param_path
    from ludwigviz.utils import aggregate_data
    from ludwigviz.utils import make_json_chart
    from ludwigviz.io import get_project_headers_and_rows
    from ludwigviz.io import count_replications

    topbar_dict = {'listing': config.RemoteDirs.research_data,
                   'hostname': hostname,
                   'version': ludwigviz.__version__,
                   'title': ludwigviz.__package__.capitalize()
                   }

    app.secret_key = 'ja0f09'
    app.run(port=5001, debug=namespace.debug, host='0.0.0.0')
