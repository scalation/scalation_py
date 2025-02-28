# Improving Channel-Independent Transformer via Statistical Preprocessing for Time Series Forecasting

This repository contains the code accompanying the paper *"Improving Channel-Independent Transformer via Statistical Preprocessing for Time Series Forecasting."* The project demonstrates how integrating advanced statistical preprocessing techniques can improve the performance of channel-independent transformer models in time series forecasting.

## Requirements

Before running the code, please ensure that the following Python packages are installed:

- numpy
- matplotlib
- pandas
- scikit-learn
- torch==1.11.0

You can install these dependencies via pip:

```bash
pip install numpy matplotlib pandas scikit-learn torch==1.11.0
```

## Files

- run.ipynb: The main Jupyter Notebook that contains the code to run the experiments.
- Additional source files and modules as needed.

## Usage
To run the code, open the run.ipynb notebook in Jupyter Notebook or JupyterLab. The notebook allows you to select different transformation techniques through the scale_method argument. The available options are:

"log1p": Log1p transformation.
"sqrt": Square root transformation.
"box-cox": Box-Cox transformation.
"standardscaler": StandardScaler.
"yeo-johnson": Yeo-Johnson transformation.

## Applying Differencing
If you want to apply differencing along with the transformation (for example, for log1p), adjust the following arguments:

- differencing: Set to true if you want to apply differencing.
- difforder: Set to either "first" for first differencing or "seasonal" for seasonal differencing.

For example, to use a log1p transformation with first differencing, configure your parameters as follows:
scale_method = "log1p"
differencing = True
difforder = "first"

These parameters allow you to experiment with different preprocessing techniques to study their effects on forecast accuracy.

