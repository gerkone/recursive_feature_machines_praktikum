# Experiment repository for the TUM praktikum "Analysis of new phenomena in machine learning"

> __Feature learning in neural networks and kernel machines that recursively learn features__<br>
> Adityanarayanan Radhakrishnan, Daniel Beaglehole, Parthe Pandit, Mikhail Belkin<br>
> https://arxiv.org/abs/2212.13881

### First part
[reproduction.ipynb](praktikum/reproduction.ipynb) contains the reproduction of the the __empirical results__ from the paper. The experiments aim at showing that RFMs are able to learn extremely similar features to those learned by neural networks. The reproduction focuses on 3 experiments:
1. __Key result__: RFMs and neural networks learn similar features
2. __Tabular data__: RFMs outperforms most models on tabular data
3. __Special phenomena__: Neural networks and RFMs exhibit grokking and simplicity bias

### Second part
[extensions.ipynb](praktikum/extension.ipynb) contains some proposed extensions or new experiments
1. __Feature learning__: where and how are features learned?
2. __Gradientless optimization__: Do different optimization methods have effects on the learned features?
3. __RFM__ _(possible)_ __improvements__: How could RFMs be improved?
4. __Structural constraints__ (e.g. equivariance): What happens for models which constraint the feature matrix, such as CNNs or EMLPs?