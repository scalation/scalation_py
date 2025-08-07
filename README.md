# Branch: dev_sr

This repository is a comprehensive collection of models, transformations, and experiments focused on **pandemic forecasting**, with an emphasis on **multivariate time series modeling** using both statistical and deep learning methods.

Developed as part of a Ph.D. research, this toolkit includes statistical models, neural networks, and deep learning models including **SARIMAX**, **Recurrent Neural Networks (LSTM, GRU)**, **Sequence-to-Sequence models with/without Attention**, **Transformer-based architectures**, and **Graph Neural Networks (GNNs)** for multivariate time series forecasting.

> 🔍 **Note:** All experiments, models, and implementations related to this research are maintained in the [`dev_sr`](https://github.com/scalation/scalation_py/tree/dev_sr) branch, which contains the author's individual contributions, including advanced normalization techniques, graph-based modeling, and transformers for multivariate time series.

## Main Projects

### 🔗 [A3T-GCN](https://link.springer.com/chapter/10.1007/978-3-031-44725-9_2)
Implements a modified Attention Temporal Graph Convolutional Network for COVID-19 forecasting across U.S. states using Pearson correlation and mutual information-based adaptive graphs.  
📄 *Published as:* [*Exploring the Predictive Power of Correlation and Mutual Information in A3T-GCN* (Springer, 2023)](https://link.springer.com/chapter/10.1007/978-3-031-44725-9_2)

### 🔗 [PandemicForecasting](https://link.springer.com/chapter/10.1007/978-3-031-77088-3_1)
Benchmarks 16 statistical and deep learning models for forecasting COVID-19 and ILI cases. It highlights that model effectiveness does not always align with complexity. The project evaluates real-world CDC models across early-stage and large-scale pandemic scenarios, incorporates a retraining strategy to handle scarce data, and applies targeted hyperparameter tuning to improve model generalization.
📄 *Published as:* [*How Effective are Time Series Models for Pandemic Forecasting?* (Springer, 2024)](https://link.springer.com/chapter/10.1007/978-3-031-77088-3_1)

### 🔬 Transformations (Ongoing)
This topic focuses on integrating statistical preprocessing methods and adaptive reversible instance normalizations into deep learning pipelines. It includes statistical and adaptive normalization, skew-aware transformations, and covariance-based scaling for multivariate time series to enhance the robustness and generalization of Transformer models in time series forecasting.


## Citation

If you use this work in your research, please consider citing the corresponding papers.
```
@inproceedings{rana2023exploring,
  title={Exploring the predictive power of correlation and mutual information in attention temporal graph convolutional network for COVID-19 forecasting},
  author={Rana, Subas and Barna, Nasid Habib and Miller, John A},
  booktitle={International Conference on Big Data},
  pages={18--33},
  year={2023},
  organization={Springer}
}
```
```
@inproceedings{rana2024effective,
  title={How Effective are Time Series Models for Pandemic Forecasting?},
  author={Rana, Subas and Miller, John A and Nesbit, John and Barna, Nasid Habib and Aldosari, Mohammed and Arpinar, Ismailcem Budak},
  booktitle={International Conference on Big Data},
  pages={3--17},
  year={2024},
  organization={Springer}
}
```
---

## Contact

For questions or collaboration inquiries, feel free to reach out:

**Subas Rana**  
Ph.D. Candidate  
Graduate Research Assistant  
University of Georgia  
📧 subas.rana187@gmail.com
📧 subas.rana@uga.edu

---


