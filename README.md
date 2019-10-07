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

## TODO

* confidence-interval
* add param2val - so that user can see params
* delete some runs
* use two_group_btn_names
* use any_group_btn_names

## Technical Note
 
Requires Python >= 3.5.3 (due to altair dependency)

If network connection is not available, 
development is possible using the `--dummy` flag:

```bash
python app.py --dummy
```

This tells the application to load csv files from a dummy location.