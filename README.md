# Time Series and Optimal-Stopping-Theory-Models Anlysis

This is my 5th Year Project for MSci Software Engineering.

## Project Description

In this project we are carrying out a decision making process for task offloading in MEC server non-sequentially using Optimal Stopping Theory models. We are designing 3 algorithms to work with:

- Random(P) Model
- Secretary Model
- House Selling Model

and we optimize them to bring the decision making process as closer as possible to the Optimal Solution. The performance of our models is evaluated using graphical representation which is the main output of our executables. In addition we provide another python module to carry out Time-Series Analysis. Our software takes in ```.csv``` files. Sample data for marker is provided in the ```\data``` folder in the source repository.

## Executables

We have two main executables. ```simulation_tool.py``` carries out the simulations for the desired model. ```time-series_analysis.py``` is simpler and carries out a times series analysis using the ```.csv``` inputted. They both operate through the command prompt.

```
simulation_tool.py
time-series_analysis.py
```

# Building/Running Instructions for Marker

These executables were built using ipython module.
To run open a command prompt window in the directory and run:

```bash
$ ipython simulation_tool.py
```
```bash
$ ipython time-series_analysis.py
```
In case there is a change in the code you can convert the .ipynbs to scripts by running the following command:

```bash
$ ipython nbconvert --to script thescript.ipynb
```

Follow the instructions on the screen when you run the executables.

In case you don't have ipython you can install it with:

```bash
$ pip3 install ipython
```

# Author
```
Author: Odysseas Polycarpou
GUID: 2210049p
```
