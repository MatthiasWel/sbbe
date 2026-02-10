## Installation

To install sbbe from GitHub repository, do:

```console
git clone git@github.com:MatthiasWel/sbbe.git
cd sbbe
conda env create -f environment.yaml
conda activate sbbe
pip install .
```

## Basic functionality

```python
from chemford.benford_mixture import BenfordMixtureEstimator

estimator = BenfordMixtureEstimator("log_BF")

# numerical values
hdi, probs, m_vals = estimator(data)

# visualization
estimator.visualize(data)

```


## Documentation

![Figure 1](assets/workflow.svg)

Simulation-Based Benfordness Estimation (SBBE) workflow. (a) SBBE requires only two inputs from a real-world dataset: the sample size n and the observed statistic (in the following logBF). The simulation produces a distribution of logBF values for each simulated Given Benfordness (the blue band in the heatmap shows the logBF distribution at Given Benfordness of 0.3). We infer Benfordness by comparing the observed logBF to simulations (red band in the heatmap), yielding a probability density over Benfordness. The points in the density plot represent the actual computed values. For improved readability, we additionally connect them with a line, which results in a curve whose integral may be less than 1. In all other plots, we show only the line.  Estimated Benfordness is the maximum a posteriori value (peak). (b) Simulated distributions of logBF for different Given Benfordness values. (c) Illustration of acquiring the distribution of logBF values for a Given Benfordness value. Specifically, logBF is calculated for 1,000 simulated datasets with sample size n and Given Benfordness 0.3.



## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
