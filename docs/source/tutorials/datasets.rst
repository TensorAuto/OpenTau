Datasets
========

Standard Data Format (for Development and Inference)
----------------------------------------------------

The "Standard Data Format" is the expected data format returned by ``torch.utils.data.Dataset``'s ``__getitem__`` and the expected input to ``torch.nn.Module``'s ``forward`` method. Any new datasets, VLMs, or VLAs that get added to this repository need to adhere to this format. Data being passed to the model during inference should also adhere to this format. The format is as follows:

.. code-block:: python

    {
        "camera0": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        "camera1": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        # ...
        "camera{num_cams-1}": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.

        "local_camera0": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        "local_camera1": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.
        # ...
        "local_camera{action_expert_num_cams-1}": torch.Tensor,  # shape (C, H, W) with values from [0, 1] and with the H, W resized to the config's specifications.

        "state": torch.Tensor,    # shape (max_state_dim)
        "actions": torch.Tensor,  # shape (action_chunk, max_action_dim)
        "frozen_actions": torch.Tensor,  # shape (frozen_actions, max_action_dim)
        "prompt": str,            # the task prompt, e.g. "Pick up the object and place it on the table."
        "response": str,          # the response from the VLM for vision QA tasks. For LeRobotDataset, this will be an empty string.
        "loss_type": str,         # the loss type to be applied to this sample (either "CE" for cross entropy or "MSE" for mean squared error)

        "img_is_pad": torch.BoolTensor,  # shape (num_cams,) with values 0 or 1, where 1 indicates that the camera image is a padded image.
        "local_img_is_pad": torch.BoolTensor,  # shape (action_expert_num_cam,) with values 0 or 1, where 1 indicates that the local camera image is a padded image.
        "action_is_pad": torch.BoolTensor,  # shape (action_chunk,) with values 0 or 1, where 1 indicates that the action is a padded action.
        "frozen_action_is_pad": torch.BoolTensor,  # shape (frozen_actions,) with values 0 or 1, where 1 indicates that the frozen action is a padded action.
    }

The config file will have to provide the following information in ``TrainPipelineConfig``:

- ``H, W``: The height and width of the camera images. These should be the same for all cameras.
- ``num_cams``: The number of cameras for the cloud VLM in the dataset.
- ``action_expert_num_cams``: The number of cameras for the action expert in the dataset.
- ``max_state_dim``: The maximum dimension of the state vector.
- ``max_action_dim``: The maximum dimension of the action vector.
- ``action_chunk``: The number of actions in the action vector. This is usually 1 for single action tasks, but can be more for multi-action tasks.

Cameras should be labeled in order of importance (e.g. camera0 is the most important camera, camera1 is the second most important camera, etc.). The model dataset will select the most important cameras to use if num_cams is less than the number of cameras in the dataset.

Both the prompt and response strings should contain exactly one newline character at the end of the string unless they are empty strings.


Computing max token length for dataset mixture
----------------------------------------------

Each training config (e.g., `dev-config <../../examples/dev_config.json>`_) should contain a dataset mixture definition. To evaluate the maximum token length for the dataset mixture, you can run the following command:

.. code-block:: bash

    python lerobot/scripts/compute_max_token_length.py \
        --target_cfg=<path/to/your/training/config.json>\
        --output_path=outputs/stats/token_count.json \
        --num_workers=10

This will output a token count for each language key in the dataset mixture, and save it to ``outputs/stats/token_count.json``.

AgiBot dataset
--------------

A clone of the ``agibot-world/AgiBotWorld-Alpha`` dataset is provided at ``/autox/teams/project-bot/AgiBotWorld-Alpha``. You can use the script ``lerobot/scripts/agibot_to_lerobot.py`` to convert it to lerobot format, saved in a given directory. For example:

.. code-block:: bash

    $ python3 lerobot/scripts/agibot_to_lerobot.py --src_path /autox/teams/project-bot/AgiBotWorld-Alpha --tgt_path <path> --task_id 327

