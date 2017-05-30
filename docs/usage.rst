=====
Usage
=====

Basic Usage
===========

Arguments for the ``diluvian`` command line interface are available via help:

.. code-block:: console

   diluvian -h
   diluvian train -h
   diluvian fill -h
   diluvian sparse-fill -h
   diluvian view -h
   ...

and also :ref:`in the section below <command-line-interface>`.


Configuration Files
-------------------

Configuration files control most of the behavior of the model, network, and
training. To create a configuration file:

.. code-block:: console

   diluvian check-config > myconfig.toml

This will output the current default configuration state into a new file.
Settings for configuration files are documented in the
:mod:`config module documentation<diluvian.config>`.
Each section in the configuration file,
like ``[training]`` (known in TOML as a *table*), corresponds with a different
configuration class:

* :class:`Volume<diluvian.config.VolumeConfig>`
* :class:`Model<diluvian.config.ModelConfig>`
* :class:`Network<diluvian.config.NetworkConfig>`
* :class:`Optimizer<diluvian.config.OptimizerConfig>`
* :class:`Training<diluvian.config.TrainingConfig>`
* :class:`Postprocessing<diluvian.config.PostprocessingConfig>`

To run diluvian using a custom config, use the ``-c`` command line argument:

.. code-block:: console

   diluvian train -c myconfig.toml

If multiple config files are provided, each will be applied on top of the
previous state in the order provided, only overriding the settings that are
specified in each file:

.. code-block:: console

   diluvian train -c myconfig1.toml -c myconfig2.toml -c myconfig3.toml

This allows easy compositing of multiple configurations, for example when
running a grid search.


Dataset Files
-------------

Volume datasets are expected to be in HDF5 files. Dataset configuration
is provided by TOML files that give the paths to these files and the HDF5
group paths to the relevant data within them.

Each dataset is a TOML array entry in the datasets table:

.. code-block:: toml

    [[dataset]]
    name = "Sample A"
    hdf5_file = "sample_A_20160501.hdf"
    image_dataset = "volumes/raw"
    label_dataset = "volumes/labels/neuron_ids"

``hdf5_file`` should include the full path to the file.

Multiple datasets can be included by providing multiple ``[[dataset]]``
sections.

To run diluvian using a dataset configuration file, use the ``-v``
command line argument:

.. code-block:: console

   diluvian train -v mydataset.toml


As a Python Library
===================

To use diluvian in a project::

    import diluvian

If you are using diluvian via Python, it most likely is because you have data
in a custom format you need to import.
The easiest way to do so is by constructing or extending the
:class:`Volume class <diluvian.volumes.Volume>`.
For out-of-memory datasets, construct a volume class backed by block-sparse
data structures (:class:`diluvian.octrees.OctreeVolume`).
See :class:`ImageStackVolume<diluvian.volumes.ImageStackVolume>` for an example.

Once data is available as a volume, normal training and filling operations can
be called. See :meth:`diluvian.training.train_network` or
:meth:`diluvian.diluvian.fill_region_with_model`.


.. _command-line-interface:

Command Line Interface
======================

.. argparse::
   :module: diluvian.__main__
   :func: _make_main_parser
   :prog: diluvian
