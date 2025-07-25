Open QBench
=====================

----------------
About project
----------------
Open QBench is a software framework designed to facilitate developing and executing benchmarks on the quantum-classical stack.

The main goal of this library is to provide a series of experiments and metrics that can be used to test the quality of different quantum devices.

----------
References
----------

- `Application Performance Benchmarks for Quantum Computers <https://arxiv.org/abs/2310.13637>`_
- `Application-Oriented Performance Benchmarks for Quantum Computing <https://arxiv.org/abs/2110.03137>`_

----------------
Getting started
----------------
For now the library is not yet hosted on PyPi so to install run

::

   pip install .

or with dependencies

::

   pip install .[VQE,QSVM,IBM,AQT,ORCA]

Then you can get to know the library by checking out the following sections:

.. toctree::
   :maxdepth: 1

   examples

   API Documentation <API/open_qbench>
