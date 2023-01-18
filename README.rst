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

  # to list all the data fields 
  print(pvqd.get_fields())

  # to list categorical fields' unique values
  print(pvqd.get_sexes()) # genders
  print(pvqd.get_natlangs()) # native languages
  print(pvqd.get_origins()) # races
  print(pvqd.get_diagnoses()) # diagnoses

  # to get a copy of the full database
  df = pvqd.query(include_diagnoses=True)

  # to get age, gender, diagnoses, and MDVP measures of non-smoking 
  # subjects with polyp or paralysis, F0 between 100 and 300 Hz
  df = pvqd.query(["AGE","SEX","DIAGNOSES","MDVP"], 
                    DIAGNOSES=["vocal fold polyp","paralysis"],
                    Fo=[100,300],
                    SMOKE=False)

  # to get the list of AH NSP files of normal subjects
  wavfiles = pvqd.get_files('ah',NORM=True)

  # to iterate over 'rainbow passage' acoustic data of female pathological subjects
  for fs, x, info in pvqd.iter_data('rainbow',
                                      auxdata_fields=["AGE","SEX"],
                                      NORM=False, SEX="F"):
    # run the acoustic data through your analysis function, get measurements
    params = my_analysis_function(fs, x)

    # log the measurements along with the age and gender info
    my_logger.log_outcome(*info, *params)
