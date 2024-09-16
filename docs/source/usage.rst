Usage
=====

Installation
------------

To use QCG-QBench, first install it using pip:

.. code-block:: console

   (.venv) $ pip install .

The VQE and QSVM benchmarks require some additional dependencies, so in order to run these, install with:

.. code-block:: console

    (.venv) $ pip install .[VQE,QSVM]

Examples
--------

.. autoexception:: qc_app_benchmarks.base_benchmark.BenchmarkError