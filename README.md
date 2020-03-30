# ST-DBSCAN

**Simple and effective method for spatial-temporal clustering**

*st_dbscan* is an open-source software package for the spatial-temporal clustering of movement data:

- Implemnted using `numpy` and `sklearn`
- Scales to memory - using chuncking see `st_dbscan.fit_frame_split`

## Installation
The easiest way to install *st_dbscan* is by using `pip` :

    pip install st-dbscan

## How to use

```python
from st_dbscan import ST_DBSCAN

st_dbscan = ST_DBSCAN(eps1 = 0.05, eps2 = 10, min_samples = 5)
st_dbscan.fit(data)

```

- __Demo Notebook:__ the following noteboook shows a demo of common features in this package -
[see Jupyter Notebook](/demo/demo.ipynb)

## Description

A package to perform the ST_DBSCAN clustering. For more details please see the following papers:

```
* Ester, M., H. P. Kriegel, J. Sander, and X. Xu, "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise". In: Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining, Portland, OR, AAAI Press, pp. 226-231. 1996
* Birant, Derya, and Alp Kut. "ST-DBSCAN: An algorithm for clustering spatial–temporal data." Data & Knowledge Engineering 60.1 (2007): 208-221.
* Peca, I., Fuchs, G., Vrotsou, K., Andrienko, N. V., & Andrienko, G. L. (2012). Scalable Cluster Analysis of Spatial Events. In EuroVA@ EuroVis.
```

## License
Released under MIT License. See the [LICENSE](LICENSE) file for details.
The package was developed by Eren Cakmak from the [Data Analysis and Visualization Group](https://www.vis.uni-konstanz.de/) and the [Department of Collective Behaviour](http://collectivebehaviour.com) at the University Konstanz funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy – EXC 2117 – 422037984“