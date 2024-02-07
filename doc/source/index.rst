.. WettingFront-LGES documentation master file, created by
   sphinx-quickstart on Mon Feb  5 23:34:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WettingFront-LGES's documentation!
=============================================

WettingFront-LGES is a plugin of WettingFront to analyze
LGES battery electrolyte wetting (anode and separator).

Installation
------------

.. code-block:: bash

   pip install git+https://gitmcpl.snu.ac.kr/lges-battery-electrolyte/wettingfront-lges.git

Examples
--------

.. _examples:

.. note::

    Before running this tutorial, environment variable ``$WETTINGFRONT_SAMPLES``
    must be set:

    .. tabs::

        .. code-tab:: bash

            export WETTINGFRONT_SAMPLES=$(wettingfront samples lges)

        .. code-tab:: bat cmd

            FOR /F %G IN ('wettingfront samples lges') DO SET WETTINGFRONT_SAMPLES=%G

        .. code-tab:: powershell

            $env:WETTINGFRONT_SAMPLES=$(wettingfront samples lges)

    Check if the variable is properly set.
    The output of ``wettingfront samples lges`` command should be same as the result of:

    .. tabs::

        .. code-tab:: bash

            echo $WETTINGFRONT_SAMPLES

        .. code-tab:: bat cmd

            echo %WETTINGFRONT_SAMPLES%

        .. code-tab:: powershell

            echo $env:WETTINGFRONT_SAMPLES

:download:`Configuration file <example.yml>`:

.. literalinclude:: example.yml
    :language: yaml

Results:

.. figure:: output/separator.jpg
   :align: center
   :figwidth: 45%

   :download:`output/separator.jpg`

.. csv-table:: :download:`output/separator.csv`
   :file: output/separator.csv
   :header-rows: 1

API reference
=============

Configuration file
------------------

.. autofunction:: wettingfront_lges.anode_analyzer

.. autofunction:: wettingfront_lges.separator_analyzer

Module reference
----------------

.. include:: autoapi/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
