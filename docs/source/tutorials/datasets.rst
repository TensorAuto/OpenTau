Datasets
========

.. note::
   Make sure you have followed the :doc:`/installation` guide before proceeding.

Adding a new dataset
--------------------




Computing max token length for dataset mixture
----------------------------------------------

Each training config (e.g., `dev-config <../../examples/dev_config.json>`_) should contain a dataset mixture definition. To evaluate the maximum token length for the dataset mixture, you can run the following command:

.. code-block:: bash

    python lerobot/scripts/compute_max_token_length.py \
        --target_cfg=<path/to/your/training/config.json>\
        --output_path=outputs/stats/token_count.json \
        --num_workers=10

This will output a token count for each language key in the dataset mixture, and save it to ``outputs/stats/token_count.json``.


