# LudwigViz


A browser interface for visualizing results (e.g. training accuracy) saved by [Ludwig](https://github.com/phueb/Ludwig), a job submission system used at the UIUC Learning & Language lab.

## Features

* View jobs submitted to Ludwig
* Visualize job results - e.g. plot performance of neural network over time

## Dependencies

* flask - the web app framework
* pandas - representing and working with tabular data
* [altair](https://altair-viz.github.io/user_guide/saving_charts.html) - a fantastic visualization API for python
* [Google Material Design Lite](https://getmdl.io/index.html) - css classes for styling

## Starting the app

Navigate to the root director, then enter:
`python -m flask run`

Pycharm uses this method, and sets `FLASK_ENV` to "development'.


## Compatibility
 
Requires Python >= 3.5.3 (due to altair dependency)


## Access to Shared Drive used by Ludwig

If access to the shared drive (owned by UIUC Learning & Language Lab) is not available, the application will load dummy data from a dummy location.
This is useful for development.
