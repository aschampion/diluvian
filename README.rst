===============================
diluvian
===============================


Flood filling networks for segmenting electron microscopy of neural tissue.

==============  ===============
PyPI Release    |pypi_badge|
Documentation   |docs_badge|
License         |license_badge|
Build Status    |travis_badge|
==============  ===============

Diluvian is an implementation and extension of the flood-filling network (FFN)
algorithm first described in [Januszewski2016]_. Flood-filling works by
starting at a seed location known to lie inside a region of interest, using a
convolutional network to predict the extent of the region within a small
field of view around that seed location, and queuing up new field of view
locations along the boundary of the current field of view that are confidently
inside the region. This process is repeated until the region has been fully
explored.


Quick Start
-----------

This assumes you already have CUDA installed and have created a fresh
virtualenv. See `installation documentation <https://diluvian.readthedocs.io/page/installation.html>`_
for detailed instructions.

Install diluvian and its dependencies into your virtualenv:

.. code-block:: console

   pip install diluvian

For compatibility diluvian only requires TensorFlow CPU by default, but you
will want to use TensorFlow GPU if you have installed CUDA:

.. code-block:: console

   pip install 'tensorflow-gpu==1.3.0'

To test that everything works train diluvian on three volumes from the
`CREMI challenge <https://cremi.org>`_:

.. code-block:: console

   diluvian train

This will automatically download the CREMI datasets to your Keras cache. Only
two epochs will run with a small sample set, so the trained model is not useful
but will verify Tensorflow is working correctly.

To train for longer, generate a diluvian config file:

.. code-block:: console

   diluvian check-config > myconfig.toml

Now edit settings in the ``[training]`` section of ``myconfig.toml`` to your
liking and begin the training again:

.. code-block:: console

   diluvian train -c myconfig.toml

For detailed command line instructions and usage from Python, see the
`usage documentation <https://diluvian.readthedocs.io/page/usage.html>`_.


Limitations, Differences, and Caveats
-------------------------------------

Diluvian may differ from the original FFN algorithm or make implementation
choices in ways pertinent to your use:

* By default diluvian uses a U-Net architecture rather than stacked convolution
  modules with skip links. The authors of the original FFN paper also now use
  both architectures (personal communication). To use a different architecture,
  change the ``factory`` setting in the ``[network]`` section of your config
  file.
* Rather than resampling training data based on the filling fraction
  :math:`f_a`, sample loss is (optionally) weighted based on the filling
  fraction.
* A FOV center's priority in the move queue is determined by the checking
  plane mask probability of the first move to queue it, rather than the
  highest mask probability with which it is added to the queue.
* Currently only processing of each FOV is done on the GPU, with movement
  being processed on the CPU and requiring copying of FOV data to host and
  back for each move.

.. [Januszewski2016]
   Michał Januszewski, Jeremy Maitin-Shepard, Peter Li, Jorgen Kornfeld,
   Winfried Denk, and Viren Jain.
   Flood-filling networks. *arXiv preprint*
   *arXiv:1611.00421*, 2016.

.. |pypi_badge|
        image:: https://img.shields.io/pypi/v/diluvian.svg
        :target: https://pypi.python.org/pypi/diluvian
        :alt: PyPI Package Version

.. |travis_badge|
        image:: https://img.shields.io/travis/aschampion/diluvian.svg
        :target: https://travis-ci.org/aschampion/diluvian
        :alt: Continuous Integration Status

.. |docs_badge|
        image:: https://readthedocs.org/projects/diluvian/badge/?version=latest
        :target: https://diluvian.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. |license_badge|
        image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://opensource.org/licenses/MIT
        :alt: License: MIT
