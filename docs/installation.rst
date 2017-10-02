.. highlight:: shell

============
Installation
============

Diluvian requires CUDA. For help installing CUDA, follow the
`TensorFlow installation <https://www.tensorflow.org/install/>`_ instructions
for GPU support.
Note that diluvian will only install TensorFlow CPU during setup, so you will
want to install the version of ``tensorflow-gpu`` diluvian requires:

.. code-block:: console

    pip install 'tensorflow-gpu==1.3.0'

You should install diluvian
`in a virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_
or similar isolated environment. All other documentation here assumes a
a virtualenv has been created and is active.

The neuroglancer PyPI package release is out-of-date, so to avoid spurious
console output and other issues you may want to
`install from source <https://github.com/google/neuroglancer/tree/master/python>`_.

To use skeletonization you must install the
`skeletopyze <https://github.com/funkey/skeletopyze>`_ library into the
virtualenv manually. See its documentation for requirements and instructions.


Stable release
--------------

To install diluvian, run this command in your terminal:

.. code-block:: console

    pip install diluvian

This is the preferred method to install diluvian, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for diluvian can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    git clone git://github.com/aschampion/diluvian

Or download the `tarball`_:

.. code-block:: console

    curl  -OL https://github.com/aschampion/diluvian/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    python setup.py install


.. _Github repo: https://github.com/aschampion/diluvian
.. _tarball: https://github.com/aschampion/diluvian/tarball/master
