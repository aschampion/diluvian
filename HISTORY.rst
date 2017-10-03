=======
History
=======

0.0.5 (2017-10-03)
------------------

* Fix bug creating U-net with far too few channels.
* Fix bug causing revisit of seed position.
* Fix bug breaking sparse fill.


0.0.4 (2017-10-02)
------------------

* Much faster, more reliable training and validation.
* U-net supports valid padding mode and other features from original
  specification.
* Add artifact augmentation.
* More efficient subvolume sampling.
* Many other changes.


0.0.3 (2017-06-04)
------------------

* Training now works in Python 3.
* Multi-GPU filling: filling will now use the same number of processes and
  GPUs specified by ``training.num_gpus``.


0.0.2 (2017-05-22)
------------------

* Attempt to fix PyPI configuration file packaging.


0.0.1 (2017-05-22)
------------------

* First release on PyPI.
