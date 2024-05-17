# Sampling Based Methods for Inner Product Sketching

This is the code for the paper "Sampling Based Methods for Inner Product Sketching" submitted to VLDB 2024.
The extended version of the paper is available at: 

- https://arxiv.org/abs/2309.16157

We suggest users read the paper for a better understanding of the experiments before using the code.

## Contents

This README file is divided into the following sections:

* [1. Requirements](#-1-requirements)
* [2. Setup before reproducing the plots](#-2-setup-before-reproducing-the-plots)
* [3. Reproducing the experimental results](#-3-reproducing-the-plots-by-running-experiments)

## 🚀 1. Requirements
The paper experiments were run using `Python 3.9.9` with the following required packages. They are also listed in the `requirements.txt` file.
- matplotlib==3.7.2
- numba==0.57.1
- numpy==1.24.4
- pandas==2.0.3
- scipy==1.11.1
- statsmodels==0.14.0
- sklearn==1.4.1

The instructions assume a Unix-like operating system (Linux or MacOS). You may need to adjust the steps for machines running Windows.

## 🚀 2. Setup before reproducing the plots

### 🔥 2.1 Create a virtual environment (optional, but recommended)

To isolate dependencies and avoid library conflicts with your local environment, you may want to use a Python virtual environment manager. To do so, you should run the following commands to create and activate the virtual environment:
```bash
python -m venv ./venv
source ./venv/bin/activate
```

### 🔥 2.2 Make sure you have the required packages installed

You can install the dependencies using `pip`:
```
pip install -r requirements.txt
```

### 🔥 2.3 Set correct environment variables PROJECT_PATH and SCRIPT_PATH by running:

```bash
source .bashrc
```

To verify that this worked, you can run `echo $PROJECT_PATH` and confirm that the output points to the directory where the repositoy was downloaded.

## 🚀 3. Reproducing the experimental results

### 🔥 3.1 Make sure you have done the [Setup](#-2-setup-before-reproducing-the-plots).

### 🔥 3.2 Use the command line to run the script with the appropriate mode.

### 🔥 3.3 Following are instructions to reproduce the experiments needed for each figure in the paper. Each subsection below describes the following points:
- explanation of the experiment
- command to run the experiment
- expected time to run the experiment based on the machine used to run the experiments: 
  - `MacBook Pro (15-inch, 2019)`
  - `2.3 GHz 8-Core Intel Core i9` with `16GB` RAM

#### ☁️ Figure 3: Inner product estimation for synthetic *real* data.
- Command: `python super_script.py -mode=ip`
- Expected time: 
  - 3.5 hours per plot
  - 14 hours for all 4 plots in Figure 3

#### ☁️ Figure 4: Inner product estimation for synthetic *binary* data. This can be applied to problems like join size estimation for tables with unique keys and set intersection estimation.
- Command: `python super_script.py -mode=join_size`
- Expected time: 
  - 1.8 hours per plot
  - 7.2 hours for all 4 plots in Figure 4

#### ☁️ Figure 5: Comparison of End-Biased Sampling (TS-1norm) and its Priority Sampling counterpart (PS-1norm) against our TS-weighted and PS-weighted methods
- Command: `python super_script.py -mode=1normVS2norm`
- Expected time: 
  - 16min per plot
  - 64min for all 4 plots in Figure 5

#### ☁️ Figure 6: Join-Correlation estimation for synthetic data.
- Command: `python super_script.py -mode=corr`
- Expected time: 
  - 7 hours per plot
  - 28 hours for all 4 plots in Figure 6

#### ☁️ Figure 7: Sketch construction time. Based on the equipment used to run the experiments, you may not be able to reproduce the exact time. However, you can still see a similar trend in the time taken by each method.
- Command: `python super_script.py -mode=time`
- Expected time: 
  - 3.5 hours for the plot

### Note that for following real data experiments, depending on the seed and samples, the results may vary slightly. However, the trend will be similar.

#### ☁️ Figure 8 and Table 2:  Inner product, correlation, and join size estimations for the World Bank data,
- Command: `python super_script.py -mode=wbf`
- Expected time: 
  - 6 hours for the figure and CSVs

#### ☁️ Figure 9:  Text similarity estimation using the 20 Newsgroups dataset
- Command: `python super_script.py -mode=20news`
- Expected time: 
  - 2 hours

#### ☁️ Figure 10:  Join size estimation for the Twitter and TPC-H datasets.
- Skewed TPC-H dataset
  - Command: `python super_script.py -mode=tpch`
  - Expected time: 
    - 2 hours
- Twitter dataset
  - Command: `python super_script.py -mode=twitter`
  - Expected time: 
    - 8 hours


### 🔥 3.4 Viewing the figures:
The figures are generated in PDF format under the directory `/fig`.