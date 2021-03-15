# Time Series Analysis and Data Quality Aware Task Offloading using Optimal Stopping Theory Models

This is my 5th Year Project for MSci Software Engineering.

## Project Description

In this project we are carrying out a decision making process for task offloading on MEC servers non-sequentially using Optimal Stopping Theory models. We are designed 3 algorithms to work with:

- Random(P) Model
- Secretary Model
- House Selling Model

We optimize the abovementioned algorithms to bring the decision making process as close as possible to the Optimal Solution. The performance of our models is evaluated using graphical representation which is the main output of our executables. In addition we are providing ```.csv``` files with the numerical results. In addition, we provide another python module to carry out Time-Series Analysis. Both executables take in ```.csv``` files as inputs. Sample data for marker is provided in the ```\build\data``` directory folder.

## Executables

We have two executables.

```
simulation_tool.py
time-series_analysis.py
```

```simulation_tool.py``` carries out the simulations for the desired model. ```time-series_analysis.py``` is simpler and carries out a times series analysis using the ```.csv``` inputted. They both operate through the command prompt.

# Building/Running Instructions for Marker

These executables were built using ipython module.
To run open a command prompt window in the directory and run:

```bash
$ ipython simulation_tool.py
```
```bash
$ ipython time-series_analysis.py
```
Each time an executable runs it produces a console output which accepts user input. An example is shown below where the user runs the ```simulation_tool.py```.

Follow the instructions on the screen.

```bash
$ ipython simulation_tool.py
Please enter the name of the .csv file you want to view: 4
                       X[t]   Y[t]
date
2014-02-14 14:32:00  44.508 -7.338
2014-02-14 14:37:00  41.244 -3.264
2014-02-14 14:42:00  48.568  7.324
2014-02-14 14:47:00  46.714 -1.854
2014-02-14 14:52:00  44.986 -1.728
...                     ...    ...
2014-02-28 14:02:00  38.474 -1.404
2014-02-28 14:07:00  40.352  1.878
2014-02-28 14:12:00  37.912 -2.440
2014-02-28 14:17:00  38.458  0.546
2014-02-28 14:22:00  37.718 -0.740

[4031 rows x 2 columns]
You can choose from:
 1 = Random(P) Model
 2 = Secretary Model
 3 = House Selling Model
 4 = Average of Models
Enter your selection: 1
Please enter the number of chunks you want to analyze. You can choose from [20,5
0,80,100,150,200]: 200
Please enter the probability you want.


You can choose from:
 0 = 0.05
 1 = 0.1
 2 = 0.2
 3 = 0.3
 4 = 0.5

Enter your selection: 1
    Run  Optimal  Load when Offloading  Load Difference
1     1   39.870                43.544            3.674
2     2   39.554                49.366            9.812
3     3   38.522                52.044           13.522
4     4   39.648                52.516           12.868
5     5   39.554                51.690           12.136
6     6   39.788                44.906            5.118
7     7   39.340                40.560            1.220
8     8   38.408                40.586            2.178
9     9   38.270                48.466           10.196
10   10   38.486                43.524            5.038
11   11   38.428                41.650            3.222
12   12   38.404                48.826           10.422
13   13   37.276                48.462           11.186
14   14   38.564                45.516            6.952
15   15   34.766                42.348            7.582
16   16   35.310                37.700            2.390
17   17   35.278                38.756            3.478
18   18   35.596                38.662            3.066
19   19   35.376                37.594            2.218
20   20   36.526                39.672            3.146

Your result figures have been saved. You can view them in the /randomp_figures/ folder!


Do you want to repeat? If not type 'exit' or 'N' to go back. Else enter 'Y' to continue:
```

# Installing Dependencies

In case there is a change in the code you can convert the .ipynbs to scripts by running the following command:

```bash
$ ipython nbconvert --to script thescript.ipynb
```

In case you don't have ```ipython``` you can install it with:

```bash
$ pip3 install ipython
```

# Author
```
Author: Odysseas Polycarpou
GUID: 2210049p
```
