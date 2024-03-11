# Sampling Based Methods for Inner Product Sketching

This is the code for the paper "Sampling Based Methods for Inner Product Sketching" submitted to VLDB 2024.
The extended version of the paper is available at: 

- https://arxiv.org/abs/2309.16157

We suggest users to read the paper for a better understanding of the experiments before using the code.

## Contents

This README file is divided into the following sections:

* [1. Requirements](#🚀-1-requirements)
* [2. Setup before reproducing the plots](#🚀-2-setup-before-reproducing-the-plots)
* [3. Reproducing the plots by running experiments](#🚀-3-reproducing-the-plots-by-running-experiments)
* [4. Reproducing the plots directly from saved data](#🚀-4-reproducing-the-plots-directly-from-saved-data)

## 🚀 1. Requirements
These experiments were run using `Python 3.9.9` with these required package. They are also listed in the `requirements.txt` file.
- matplotlib==3.7.2
- numba==0.57.1
- numpy==1.24.4
- pandas==2.0.3
- scipy==1.11.1
- statsmodels==0.14.0

## 🚀 2. Setup before reproducing the plots

### 🔥 2.1 make sure you have the required packages installed

### 🔥 2.2 set correct environment variables PROJECT_PATH and SCRIPT_PATH by running:
```bash
source .bashrc
```

## 🚀 3. Reproducing the plots by running experiments

### 🔥 3.1 make sure you have done the [Setup](#🚀-setup-before-reproducing-the-plots)

### 🔥 3.2 use the command line to run the script with the appropriate mode.

### 🔥 3.3 following are instructions to reproduce the plots by running experiments, for each part it has following parts:
- explanation of the experiment
- command to run the experiment
- expected time to run the experiment based on the machine used to run the experiments: 
  - `MacBook Pro (15-inch, 2019)`
  - `2.3 GHz 8-Core Intel Core i9` with `16GB` RAM

#### ☁️ Figure 3: Inner product estimation for synthetic *real* data.
- Command: `python super_script.py -mode=ip`
- Expected time: 
  - 3.5 hour per plot
  - 14 hours for all 4 plots in Figure 3

#### ☁️ Figure 4: Inner product estimation for synthetic *binary* data. This can be applied to problems like join size estimation for tables with unique keys and set intersection estimation.
- Command: `python super_script.py -mode=join_size`
- Expected time: 
  - 1.8 hour per plot
  - 7.2 hours for all 4 plots in Figure 4

#### ☁️ Figure 5: Comparison of End-Biased Sampling (TS-1norm) and its Priority Sampling counterpart (PS-1norm) against our TS-weighted and PS-weighted methods
- Command: `python super_script.py -mode=1normVS2norm`
- Expected time: 
  - 16min per plot
  - 64min for all 4 plots in Figure 5

#### ☁️ Figure 6: Join-Correlation estimation for synthetic data.
- Command: `python super_script.py -mode=corr`
- Expected time: 
  - 7 hour per plot
  - 28 hours for all 4 plots in Figure 6

#### ☁️ Figure 7: Sketch construction time. Based on the equipment used to run the experiments, you may not be able to reproduce the exact time. However, you can still see the similar trend in the time taken by each method.
- Command: `python super_script.py -mode=time`
- Expected time: 
  - 3.5 hour for the plot

### Note that for following real data experiments, depending on the seed and samples, the results may vary slightly. However, the trend will be similar.

#### ☁️ Figure 8 and Table 2:  Inner product, correlation, and join size estimations for the World Bank data,
- Command: `python super_script.py -mode=wbf`
- Expected time: 
  - 6 hour for the figure and CSVs

#### ☁️ Figure 9:  Text similarity estimation using the 20 Newsgroups dataset
- Command: `python super_script.py -mode=20news`
- Expected time: 
  - 2 hour

#### ☁️ Figure 10:  Join size estimation for the Twitter and TPC-H datasets.
- Skewed TPC-H dataset
  - Command: `python super_script.py -mode=tpch`
  - Expected time: 
    - 2 hour
- Twitter dataset
  - Command: `python super_script.py -mode=twitter`
  - Expected time: 
    - 8 hour

## 🚀 4. Reproducing the plots directly from saved data
To reproduce the experiments results, you can run the following command.

#### 🔥 use the command line to run the script with the appropriate figure number. For example, to generate Figure 3, run:
```bash
python experiment_plot.py --paper_fig 3
```
To generate Figure 4, run:
```bash
pyhon experiment_plot.py --paper_fig 4
```
And so on, except for the World Bank Experiemnts (Figure 8 and Table 2). Because the figure is in a different format and the table is easier to show in a jupyter notebook. To generate these, run:
```bash
WorldBankExperiment_Figure8_Table2.ipynb
```
#### 🔥 The generated figures will be saved in the folder `fig/`

## 🚀 Running your own experiments

