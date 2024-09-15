## Ordinal Forests

Implementation of Random Forests for ordinal regression as described in Ordinal Forests (2019)<sup>[1],[2]</sup> in Python. In addition, an ordinal-specific loss function, the All-Thresholds Loss, is implemented (motivated by the Loss Function (All-Threshold Loss) for ordinal response variables<sup>[3]</sup>) as the z-scoring inherently being done by Ordinal Forests provide convenient thresholding for this loss.


Based on scikit-learn version 1.1.2<sup>[4]</sup>.

Sources:

[1] https://link.springer.com/article/10.1007/s00357-018-9302-x

[2] https://cran.r-project.org/web/packages/ordinalForest/index.html

[3] https://github.com/scikit-learn/scikit-learn/issues/16694

[4] https://github.com/scikit-learn/scikit-learn/tree/1.1.2
