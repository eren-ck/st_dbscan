# ST-DBSCAN

**Simple and effective method for spatial-temporal clustering**

*st_dbscan* is an open-source software package for the spatial-temporal clustering of movement data:

- Implemnted using `numpy` and `sklearn`
- Scales to memory - using chuncking sparse matrices and the `st_dbscan.fit_frame_split`

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

A package to perform the ST_DBSCAN clustering. If you use the package, please consider citing the following benchmark paper:

```bibtex
@inproceedings{cakmak2021spatio,
        author = {Cakmak, Eren and Plank, Manuel and Calovi, Daniel S. and Jordan, Alex and Keim, Daniel},
        title = {Spatio-Temporal Clustering Benchmark for Collective Animal Behavior},
        year = {2021},
        isbn = {9781450391221},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3486637.3489487},
        doi = {10.1145/3486637.3489487},
        booktitle = {Proceedings of the 1st ACM SIGSPATIAL International Workshop on Animal Movement Ecology and Human Mobility},
        pages = {5–8},
        numpages = {4},
        location = {Beijing, China},
        series = {HANIMOB '21}
}
```

## License
Released under MIT License. See the [LICENSE](LICENSE) file for details.
The package was developed by Eren Cakmak from the [Data Analysis and Visualization Group](https://www.vis.uni-konstanz.de/) and the [Department of Collective Behaviour](http://collectivebehaviour.com) at the University Konstanz funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy – EXC 2117 – 422037984“