# Sampling Based Methods for Inner Product Sketching

This is the code for the paper "Sampling Based Methods for Inner Product Sketching" submitted to VLDB 2024.

## 🚀 Requirements
- matplotlib==3.7.
- numba==0.57.1
- numpy==1.24.4
- pandas==2.0.3
- scipy==1.11.1
- statsmodels==0.14.0

## 🚀 Reproducing the experiments results
To reproduce the experiments results, you can run the following command.

#### 🔥 1. make sure you have the required packages installed

#### 🔥 2. set correct environment variables PROJECT_PATH and SCRIPT_PATH by running:
```bash
source .bashrc
```

#### 🔥 3. use the command line to run the script with the appropriate figure number. For example, to generate Figure 3, run:
```bash
python experiment_plot.py --paper_fig 3
```
To generate Figure 4, run:
```bash
hon experiment_plot.py --paper_fig 4
```
And so on, except for the World Bank Experiemnts (Figure 8 and Table 2). Because the figure is in a different format and the table is easier to show in a jupyter notebook. To generate these, run:
```bash
WorldBankExperiment_Figure8_Table2.ipynb
```
#### 🔥 4. The generated figures will be saved in the folder `fig/`

## 🚀 Running your own experiments
To run your own experiments, you can use the following command with differnt modes:
- ip: Inner Product
- corr: Correlation
- join_size: Join Size

For example, to run the Inner Product experiment, run:
```bash
python super_script.py -mode=ip
```