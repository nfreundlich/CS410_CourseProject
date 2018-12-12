# GFLM: mine implicit features using a generative feature language model

## Description
This package implements a Generative Feature Language Models for Mining Implicit Features.

#### Given the following input:
* a text dataset
* a set of predefined features

#### Compute the following:
* mapping of explicit and implicit features on the data
* using both gflm_word and gflm_section algorithms

## Install
```
pip install feature_mining
```

## Sample Usage
    Usage:
        from feature_mining import FeatureMining
        fm = FeatureMining()
        fm.load_ipod(full_set=False)
        fm.fit()
        fm.predict()
    
    Results:
        - prediction using 'section': fm.gflm.gflm_section
        - prediction using 'word': fm.gflm.gflm_word

## Package created based on the following paper
S. Karmaker Santu, P. Sondhi and C. Zhai, "Generative Feature Language Models for Mining Implicit Features from Customer Reviews", Proceedings of the 25th ACM International on Conference on Information and Knowledge Management - CIKM '16, 2016.

## Pydocs (Code Documentation)
Accessible via this link: http://htmlpreview.github.io/?https://github.com/nfreundlich/CS410_CourseProject/blob/hannah/docs/feature_mining.html

## Tutorial
See Jupyter notebook tutorial https://github.com/nfreundlich/CS410_CourseProject/blob/dev/tutorial.ipynb
