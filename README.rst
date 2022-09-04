****************************************************************
adasamp: adaptive bayes-sampling for multi-criteria optimization
****************************************************************

Adaptive sampling: Adaptive optimization algorithm for black-box multi-objective optimization problems with binary constraints on the foundation of Bayes optimization.

**Contents**

+ sampling.py: Main file containing the algorithm.
+ models.py: Helper classes for adaptive sampling models based on scikit-learn.

**Dependencies**

+ NumPy 1.19.1
+ SciPy 1.5.2
+ pathos 0.2.6 (optional, required for parallel computing)

📖 **Citation**

The algorithm is an implementation from our paper "Adaptive Sampling of Pareto Frontiers with Binary Constraints Using Regression and Classification". Preprint available on [arXiv](https://arxiv.org/abs/2008.12005). If you find this code useful in your research, please consider citing:

.. code-block::

    @misc{heesebortzCITE2020,
		  title={Adaptive Sampling of Pareto Frontiers with Binary Constraints Using Regression and Classification}, 
		  author={Raoul Heese and Michael Bortz},
		  year={2020},
		  eprint={2008.12005},
		  archivePrefix={arXiv},
		  primaryClass={stat.ML}
         }
	