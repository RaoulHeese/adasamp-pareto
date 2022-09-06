***********************************************************************
adasamp-pareto: adaptive bayes-sampling for multi-criteria optimization
***********************************************************************

.. image:: https://github.com/RaoulHeese/adasamp-pareto/actions/workflows/tests.yml/badge.svg 
    :target: https://github.com/RaoulHeese/adasamp-pareto/actions/workflows/tests.yml
    :alt: GitHub Actions
	
.. image:: https://readthedocs.org/projects/adasamp-pareto/badge/?version=latest
    :target: https://adasamp-pareto.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status	
	
.. image:: https://img.shields.io/badge/license-MIT-lightgrey
    :target: https://github.com/RaoulHeese/adasamp-pareto/blob/main/LICENSE
    :alt: MIT License	

Adaptive optimization algorithm for black-box multi-objective optimization problems with binary constraints on the foundation of Bayes optimization. The algorithm aims to find the Pareto-optimal solution of

.. math::

   \operatorname{max} \mathbf{y}(\mathbf{x}) \hspace{2cm} \\
   \operatorname{s.t.} f(\mathbf{x}) = \text{feasible}
   
in an iterative procedure. Here, :math:`\mathbf{y}(\mathbf{x})` denotes the multi-dimensional goals and :math:`f(\mathbf{x})` the binary feasibility of the problem (in the sense that certain design variables :math:`\mathbf{x}` lead to invalid goals). All technical details can be found in the paper "Adaptive Sampling of Pareto Frontiers with Binary Constraints Using Regression and Classification" (`<https://arxiv.org/abs/2008.12005>`_).

**Installation**

Install via ``pip`` or clone this repository. In order to use ``pip``, type:

.. code-block:: sh

    $ pip install adasamp
	
**Usage**

The class ``AdaptiveSampler`` is used to define and solve a problem instance. Simple example:

.. code-block:: python

  from adasamp import AdaptiveSampler

  # Create instance
  sampler = AdaptiveSampler(func,       # Problem definition: function returns (goals Y, feasibility f)
                            X_limits,   # Design variable limits to search solution in
                            Y_ref,      # Reference point, has to be dominated by any goal Y
                            iterations, # Number of solver iterations
                            Y_model,    # Regression model to predict goals Y
                            f_model)    # Classification model to predict feasibility f

  # Return the sampling suggestions X, the corresponding goals Y, and the corresponding feasibilities f.
  X, Y, f = sampler.sample()
  
Demo notebooks can be found in the `examples/` directory.
  
**Documentation**

Complete documentation is available: `<https://adasamp-pareto.readthedocs.io/en/latest>`_.

📖 **Citation**

If you find this code useful in your research, please consider citing:

.. code-block::

    @misc{heesebortzCITE2020,
		  title={Adaptive Sampling of Pareto Frontiers with Binary Constraints Using Regression and Classification}, 
		  author={Raoul Heese and Michael Bortz},
		  year={2020},
		  eprint={2008.12005},
		  archivePrefix={arXiv},
		  primaryClass={stat.ML}
         }
	