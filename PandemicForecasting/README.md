# How Effective are Time Series Models for Pandemic Forecasting?

This repository contains the code and resources for the paper "How Effective are Time Series Models for Pandemic Forecasting?". This research explores the effectiveness of various time series models in predicting pandemics. 

## Repository Structure
The repository is organized into three main directories corresponding to the different experimental setups used in the paper:
- `Train-Test split/`: Contains the code and data for experiments using a traditional train-test split approach.
- `RollingWindow/`: Contains the code and data for experiments using a rolling window approach.
- `CDC_Experiments/`: Contains the code and data for experiments conducted using CDC data.

## Getting Started
Our project comprises various machine learning models, including both neural networks and transformer-based models, as well as a statistical model implemented using a Seasonal AutoRegressive Integrated Moving Average with exogenous variables (SARIMAX). Each model has a separate codebase organized with a main script (main.py) for running experiments and a Jupyter notebook for the SARIMAX model.

### Machine Learning Models
- Feed Forward Neural Network (FFNN)
- Gated Recurrent Unit (GRU)
- Long Short Term Memory (LSTM)
- GRU Sequence-to-Sequence
- LSTM Sequence-to-Sequence
- GRU Sequence-to-Sequence with Attention
- LSTM Sequence-to-Sequence with Attention
- DLinear
- Nlinear
### Transformer Based Models
- Transformer
- Informer
- Autoformer
- FEDformer
- PatchTST
### Statistical Model
- RandomWalk
- SARIMAX
