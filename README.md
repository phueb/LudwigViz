# LudwigViz


A browser interface for [Ludwig](https://github.com/phueb/Ludwig), a job submission system used at the UIUC Learning & Language lab.

## Features

* View jobs submitted to Ludwig
* Visualize job results - e.g. plot performance of neural network over time

## Dependencies

* flask - the web app framework
* pandas - representing and working with tabular data
* [altair](https://altair-viz.github.io/user_guide/saving_charts.html) - a fantastic visualization API for python
* [Google Material Design Lite](https://getmdl.io/index.html) - css classes for styling

## Starting the app

A flask app can be run in at least 2 ways:
1. `python -m flask run`
2. `python app.py `

Pycharm uses method 1, and sets `FLASK_ENV` to "development'.
Any `app.run()` call is ignored. 

Method 2 is preferred because it allows custom argument parsing. 
By default this results in `FLASK_ENV` set to "production". 

## Technicalities

### Compatibility
 
Requires Python >= 3.5.3 (due to altair dependency)


### Network Connection to Ludwig

If network connection is not available, 
development is possible using the `--dummy` flag:

```bash
python app.py --dummy
```

This tells the application to load csv files from a dummy location.

## TODO

* plotting of confidence-interval
* implement deleting of data associated with runs
* "compare" button is not used