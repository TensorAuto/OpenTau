Setup and Configuration
=======================

AutoX Installation
------------------

Download the source code:

.. code-block:: bash

    $ git clone git@code.autox.ds:xisp/agi/lerobot.git
    $ cd lerobot

We use `uv <https://docs.astral.sh/uv/>`_ to manage Python dependencies as it is easier and faster to use than Conda. If you would still like to use Conda, see the Conda environment setup below in the LeRobot installation instructions. See the `uv installation instructions <https://docs.astral.sh/uv/getting-started/installation/>`_ to set it up. Once uv is installed, run the following to set up the environment:

If you haven't already, please install cmake 3.x using ``apt``, ``brew``, or another package manager.

.. code-block:: bash

    $ uv sync --extra tau0 --extra test --extra video_benchmark --extra accelerate --extra dev --extra feetech --extra openai --extra onnx --extra smolvla --extra libero
    $ source .venv/bin/activate

Note that PI0.5 and Tau0/PI0 are not compatible with each other due to different ``transformers`` package versions. If you need to run PI0.5, run the following command which will override the ``transformers`` package version:

.. code-block:: bash

    $ uv pip install -r requirements-pi05.txt

To use `Weights and Biases <https://docs.wandb.ai/quickstart>`_ for experiment tracking, log in with:

.. code-block:: bash

    $ wandb login

Setting Pre-commit
------------------

Please, always set pre-commit, so it will check code styling and other necessary environment checks before committing to git.

Check if ``.pre-commit-config.yaml`` file is present under lerobot directory.
To set pre-commit use the following command:

.. code-block:: bash

    pre-commit install

After successfully executing the above command, the pre-commit checks will automatically be done whenever git commit is called. If any error is found by pre-commit, the commit to git will be made only when the error is fixed.

Setting .ENV file
-----------------

To run high level planner with gpt4o, OpenAI api key is needed to be set. Create an ``.env`` file under lerobot directory and set the variable ``OPENAI_API_KEY`` to your openai api key. The high level planner inference script will automatically load the api key and pass to openai client.

