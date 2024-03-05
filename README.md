# sutd-brain-comp-jan-24-team-project

This is code modified for a class project

## "Unsupervised Learning of Digit recognition using STDP" in Brian2

This is a modified version of the source code for the paper:

‘Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity’, Diehl and Cook, (2015).

Original code: Peter U. Diehl (https://github.com/peter-u-diehl/stdp-mnist)

Updated for Brian2: zxzhijia (https://github.com/zxzhijia/Brian2STDPMNIST)

Updated for Python3: sdpenguin (https://github.com/sdpenguin/Brian2STDPMNIST)

## Prerequisites

1. Brian2 - can be installed via pip requirements as mentioned below
2. MNIST datasets - needs to be downloaded, and point the variable `MNIST_data_path` on line 26 of Backup_Brain2_feb24.py to it

# Using Command Line Only
This is faster way to get up and running, but developer unfriendly in long-run. For better code management, please see next section with VSCode

Open windows termnial in the project folder. And then input command:

`> pip install -r requirements.txt`

> Caution: this will install all the modules in above file in your globally accessable python. It could potentially cause conflict with items that have different version on your system. This is mainly recommended to get it working right away, and specially if you have a relatively fresh copy of python installed.

Then to run the project. In the command prompt, input:

`> python .\Backup_Brian2_feb24.py`

# Instructions to Setup venv (local dedicated dev env) in Visual Studio Code

Guide for setting up venv:
https://code.visualstudio.com/docs/python/python-tutorial

## TLDR

Open the Command Palette (Ctrl+Shift+P), start typing the Python: Create Environment command to search, and then select the command.

`Select venv`

## Select The Interpreter
Ensure your new environment is selected by using the Python: Select Interpreter command from the Command Palette.

In case venv already exists, go through Create Enviornment and VScode will offer to use the existing enviornment.

## Activate venv
Check venv is working, will show a green (venv) at start of command line Open terminal and type following:

``` 
> .venv\Scripts\activate
> where python
> py -m pip --version
```

If a requirements file exists, then use it for install

`> py -m pip install -r requirements.txt` 

Or install module manually

`> py -m pip install requests`

