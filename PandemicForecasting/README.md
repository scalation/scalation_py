# How Effective are Time Series Models for Pandemic Forecasting?

This repository contains the code and resources for the paper [**"How Effective are Time Series Models for Pandemic Forecasting?"**](https://link.springer.com/chapter/10.1007/978-3-031-77088-3_1). This research explores the effectiveness of various time series models in predicting pandemics.

## Repository Structure
The repository is organized into three main directories corresponding to the different experimental setups used in the paper:
- `Train-Test split/`: Contains the code and data for experiments using a traditional train-test split approach.
- `RollingWindow/`: Contains the code and data for experiments using a ReTraining (RT) approach.
- `CDC_Experiments/`: Contains the code and data for experiments conducted using Centers for Disease Control and Prevention (CDC) data.

## Getting Started
Our project comprises various machine learning models, including both neural networks and transformers, as well as statistical models. Each model has a separate codebase organized with a main script (**main.py**) for running experiments and a Jupyter Notebook for the Seasonal AutoRegressive Integrated Moving Average (SARIMA) model.

### Neural Networks
All the models below are built using the [torch.nn](https://pytorch.org/docs/stable/nn.html) module from PyTorch, which provides a comprehensive set of tools and functionalities to construct and train neural networks efficiently while DLinear and NLinear codes are taken from the [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) repository. 

- Feed Forward Neural Network (FFNN)
- [Gated Recurrent Unit](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU) (GRU)
- [Long Short Term Memory](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM) (LSTM)
- GRU Sequence-to-Sequence
- LSTM Sequence-to-Sequence
- GRU Sequence-to-Sequence with Attention
- LSTM Sequence-to-Sequence with Attention
- [DLinear](https://arxiv.org/abs/2205.13504)
- [NLinear](https://arxiv.org/abs/2205.13504)
  
### Transformers
- [Transformer](https://arxiv.org/abs/1706.03762)
- [Informer](https://arxiv.org/abs/2012.07436)
- [Autoformer](https://arxiv.org/abs/2106.13008)
- [FEDformer](https://arxiv.org/abs/2201.12740)
- [PatchTST](https://arxiv.org/abs/2211.14730)

Thanks to the [PatchTST](https://github.com/yuqinie98/PatchTST) repository for providing all the transformers' code in one place.
### Statistical Models
- RandomWalk
- [SARIMA](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)

The SARIMA model uses the [Statsmodel](https://www.statsmodels.org/stable/index.html) library.

### Pandemic Datasets
The datasets are located in the `datasets/` directory for each codebase.
- The COVID-19 daily data was extracted from the COVID-19 Dataset by [Our World in Data](https://github.com/owid/covid-19-data) (OWID) and converted to weekly data.
- CDC COVID-19 Weekly Deaths Forecasts are collected from the [COVID-19 Forecast Hub](https://covid19forecasthub.org/). The data has been archived now and is available on [GitHub](https://github.com/scalation/data/blob/master/CDC-COVID-Data/concatenated_CDC_20_21_22_23.csv). It includes the 4-weeks ahead deaths forecasts for the models submitted to the CDC.
- The ILI Weekly dataset is collected from the CDC, and the ILI patients' data is recorded every week in the US. It is available on [GitHub](https://github.com/scalation/data/blob/master/Influenza/national_illness.csv). It is the same dataset as used by transformers.
