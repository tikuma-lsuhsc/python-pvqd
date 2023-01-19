`pvqd`: Voice Foundation Pathological Voice Quality Database Reader module
==========================================================================

|pypi| |status| |pyver| |license|

.. |pypi| image:: https://img.shields.io/pypi/v/pvqd
  :alt: PyPI
.. |status| image:: https://img.shields.io/pypi/status/pvqd
  :alt: PyPI - Status
.. |pyver| image:: https://img.shields.io/pypi/pyversions/pvqd
  :alt: PyPI - Python Version
.. |license| image:: https://img.shields.io/github/license/tikuma-lsuhsc/python-pvqd
  :alt: GitHub

.. note::
   This Python package is still under development.

This Python module provides functions to retrieve data and information easily from 
Voice Foundation's Pathological Voice Quality Database.

This module currently does not retrieve the database itself on its own. User must
download and extract the files first from https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9dz247gnyb-2.zip

Install
-------

.. code-block:: bash

  pip install pvqd

Use
---

.. code-block:: python

  from pvqd import PVQD

  # to initialize (must call this once in every Python session)
  pvqd = PVQD('<path to the root directory of the extracted database>')

  # to get a copy of the full database with averaged CAPE-V scores
  df = pvqd.query(include_cape_v=True)

  # to get age, gender, and mean GRBAS grade scores
  df = pvqd.query(["Age", "Gender"], include_grbas='grade')

  # to get a dataframe of WAV files and start and ending timestamps of all /a/ segment
  df = pvqd.get_files('/a/')

  # to iterate over '/a/' acoustic data of female participants along with
  # age and mean GRBAS scores
  for id, fs, x, auxdata in pvqd.iter_data('/a/',
                                      auxdata_fields=["Age"],
                                      include_grbas=True,
                                      Gender="Female"):
    # run the acoustic data through your analysis function, get measurements
    params = my_analysis_function(fs, x)

    # log the measurements along with the age and GRBAS info
    my_logger.log_outcome(id, *auxdata, *params)
