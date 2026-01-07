Concepts
========

This section explains the core concepts used in the OpenTau codebase.

Policies
--------
Policies map observations (e.g., camera images, robot proprioceptive states) to actions (or action chunks).
Policies are implemented as PyTorch modules and inherit from ``opentau.policies.pretrained.PreTrainedPolicy``.

Datasets
--------
Datasets are used to handle data loading and processing.
It supports downloading datasets from the Hugging Face Hub and loading them from local disk.
The dataset format is versioned (currently v2.1) and utilizes parquet files for data and mp4 files for videos to ensure efficiency and portability.
There are currently two types of datasets:

*   ``LeRobotDataset``: For robotic data.
*   ``GroundingDataset``: For VLM datasets such as Visual Question Answering (VQA) or visual grounding.

These datasets are used to train policies.

DatasetMixture
^^^^^^^^^^^^^^
To train policies on multiple datasets simultaneously, OpenTau uses ``opentau.datasets.dataset_mixture.WeightedDatasetMixture``.
This class:

*   Combines multiple ``LeRobotDataset`` and ``GroundingDataset`` instances.
*   Different weights can be assigned to each dataset to control the sampling frequency.
*   Aggregates statistics from all constituent datasets to ensure consistent normalization across the mixture.
*   Resamples the action output frequency to match the action frequency specified in the configuration.

Metadata
^^^^^^^^
Metadata is crucial for defining the structure and statistics of a dataset. Handled by ``LeRobotDatasetMetadata`` and ``DatasetMetadata``, it includes:

*   **Info**: Feature shapes, data types, FPS, and robot type.
*   **Stats**: Mean, standard deviation, min, and max values for each feature, used for normalization (e.g., standardizing images or normalizing action vectors).
*   **Tasks**: Natural language descriptions of the tasks contained in the dataset.

Metadata is stored in JSON files (``info.json``, ``stats.json``) and JSONL files (``tasks.jsonl``) within the dataset directory.

Standard Data Format
--------------------
To ensure compatibility across different datasets and policies, OpenTau introduces the **Standard Data Format**.
The Standard Data Format is the expected data format returned by ``torch.utils.data.Dataset``'s ``__getitem__`` and the expected input to ``torch.nn.Module``'s ``forward`` method. Any new datasets, VLMs, or VLAs that get added to this repository need to adhere to this format. Data being passed to the model during inference should also adhere to this format. The format is as follows:

.. code-block:: python

    {
        "camera0": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        "camera1": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        # ...
        "camera{num_cams-1}": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.

        "state": torch.Tensor,    # shape (max_state_dim)
        "actions": torch.Tensor,  # shape (action_chunk, max_action_dim)
        "prompt": str,            # the task prompt, e.g. "Pick up the object and place it on the table."
        "response": str,          # the response from the VLM for vision QA tasks. For LeRobotDataset, this will be an empty string.
        "loss_type": str,         # the loss type to be applied to this sample (either "CE" for cross entropy or "MSE" for mean squared error)

        "img_is_pad": torch.BoolTensor,  # shape (num_cams,) with values 0 or 1, where 1 indicates that the camera image is a padded image.
        "action_is_pad": torch.BoolTensor,  # shape (action_chunk,) with values 0 or 1, where 1 indicates that the action is a padded action.
    }

The config file will have to provide the following information in TrainPipelineConfig:

*   ``H, W``: The height and width of the camera images. These should be the same for all cameras.
*   ``num_cams``: The number of cameras for the cloud VLM in the dataset.
*   ``max_state_dim``: The maximum dimension of the state vector.
*   ``max_action_dim``: The maximum dimension of the action vector.
*   ``action_chunk``: The number of actions in the action vector. This is usually 1 for single action tasks, but can be more for multi-action tasks.

Cameras should be labeled in order of importance (e.g. camera0 is the most important camera, camera1 is the second most important camera, etc.). The model dataset will select the most important cameras to use if num_cams is less than the number of cameras in the dataset.

Configs
-------
Configuration management is handled using `Draccus <https://github.com/dlwh/draccus>`_.
The main configuration class is ``opentau.configs.train.TrainPipelineConfig``, which orchestrates training settings,
policy configuration, and environment setup. Configs can be loaded from pretrained checkpoints to reproduce experiments.

Environments
------------
Environments wrap simulation or real-robot interfaces compatible with OpenAI Gym/Gymnasium.
The factory `src/opentau/envs/factory.py <https://github.com/TensorAuto/OpenTau/blob/main/src/opentau/envs/factory.py>`_ creates vectorized environments for efficient training and evaluation.
Currently, only `Libero <https://libero-project.github.io/main.html>`_ is supported and it is configured via ``opentau.envs.configs.LiberoEnv``.
