# Serkan's Rotation: CCXP and sth

This repository contains code for processing, analyzing, and visualizing neural data. The project is organized into several folders, each containing modules for specific functionalities.


## ⚠️ IMPORTANT ⚠️  
If you are **not** using macOS, delete the `pink_rigs_tool` folder in this directory and install `pink_rigs_tool` from Flora's repository instead. Otherwise, you **will not** be able to access all data from the servers.  

### Why?  
- The `pink_rigs_tool` in this repo is **modified** to look for data in a local directory.  

### Why is this necessary?  
- When connecting to a server on macOS, it is mounted as `Volumes`.  
- If you connect to **multiple** servers, they are **all**


## Folder Structure

### `class_objects`
Contains class definitions.  
- **v1**: Deprecated and no longer in use.  
- **v2**: The main class used for data processing and analysis.

### `pink_rigs_tool`
Contains Flora's code. Use this tool to load data when it is stored locally rather than on the cloud.

### `plotting`
Contains modules for visualization.  
- **`session_plotter.py`**: Contains the `SessionPlotter` class for creating various plots.  
- **`helper.py`**: Contains helper functions for plotting.

### `stats`
Contains modules for statistical analysis.  
- **`cccp_ccsp.py`**: Code and calculations for CCXP (Choice-Conditioned Cross-correlation).  
- **`test_fr_change.py`**: Code for performing paired t-tests and Wilcoxon tests to assess changes in firing rates.

### `utils`
Contains various utility modules:  
- **`pca_utils.py`**: Prepares data, applies PCA, and generates corresponding plots.  
- **`spike_utils.py`**: Formats spike data, calculates firing rates, and standardizes firing rates using the formula:  

Standardized FR:  
$$
\frac{\text{firing rate} - \text{mean}(\text{baseline firing rate})}{\text{std}(\text{baseline firing rate})}
$$

- **`trial_utils.py`**: Adds new columns to the trial DataFrame and provides functions to select specific trial types.  
- **`utils.py`**: Contains code necessary for creating the main object.

## Notebook

### `v2.ipynb`
A Jupyter Notebook that demonstrates the following:
1. **Data loading**  
2. **Object creation**  
3. **Identification of significant neurons** using either:  
 - Paired t-test / Wilcoxon test, or  
 - CCXP calculations.  
4. **Visualization of results and data**, including:  
 - PSTH  
 - Raster plots  
 - CCXP results  
5. **PCA analysis** *(Note: PCA functionality may be outdated as the data structure has changed.)*
