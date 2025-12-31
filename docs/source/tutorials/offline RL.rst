Offline RL Training on pi0 policy
=================================


Introduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This tutorial demonstrates how we train a pi0 policy on libero dataset using offline RL that closely follows the training procedure of OpenPI on pi06-star.
Currently, we only support offliner RL training on pi0 policy. The future, we will make it compatible with pi05 policy .

The procedure is as follows:

1. Pre-Train VLA policy on whole libero dataset till convergence.
2. Pre-Train value function on whole libero dataset till convergence.
3. Repeat the below steps for 1-3 times:
    i. Collect the dataset by rolling out the VLA policy on the single libero task
    ii. Fine Tune the value function on collection of original dataset and all the previously rolled out dataset
    iii. Compute the advantage for each data point using the fine-tuned value function and calculate the epsilon threshold for setting I\ :sub:`t`\ (Indicator) VLA policy training.
    iv. Fine Tune the VLA policy on collection of original dataset and all the previously rolled out dataset


Stage 1: Pre-Train VLA policy on whole libero dataset till convergence.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pi0 pretrained checkpoint is used as starting point for this step.
Pi0 is trained on whole libero dataset (physical intelligence/libero) for around 10k steps before it converges to 80% success rate on moka pot libero task.
During the whole training, \ :sub:`t`\ Indicator is set to True for all the data points, i.e assuming that the expert policy is always taking optimal action for a given state.
Training the policy on whole libero tasks gives a better baseline for the offline RL training compared to just training on a single task.


Stage 2: Pre-Train value function on whole libero dataset till convergence.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instead of training the value function on huge pre training dataset of pi05 policy, we consider training it on whole libero dataset (physical intelligence/libero) as a pre-training step.
This helps to have a good baseline for value function training in the offline RL training, avoiding overfitting by training on multiple tasks, thus providing multi task value function.
Value function was trained for approximately 80k steps to achieveing close to 100% accuracy on whole libero dataset.

Recipe for value function training:
1. Defining reward function for each action:

   The reward function is defined as follows:

   .. math::

      r_t = \begin{cases}
         0 & \text{if } t = T \text{ and success} \\
         -C_{\text{fail}} & \text{if } t = T \text{ and failure} \\
         -1 & \text{otherwise}
      \end{cases}

   where :math:`t` is the timestep, :math:`T` is the final timestep of the episode, and :math:`C_{\text{fail}}` is a large negative constant for failed episodes.

   The value function is trained to predict the negative number of remaining steps until success or a large negative value for failed episodes, with values normalized between :math:`(-1, 0)`.

2. At each timestep :math:`t` we calculate the return using the above mentioned reward function as :math:`R_t(\tau) = \sum_{t'=t}^{T} r_{t'}`.

3. The returns are then discretized into :math:`B = 201` bins using equal width binning, denoted as :math:`R^B(\tau)`.

4. The value function is trained to predict the bin index of the discretized return using cross-entropy loss. The training objective is:

   .. math::

      \min_{\phi} \mathbb{E}_{\tau \sim D} \left[ \sum_{o_t \in \tau} H(R^B(\tau), p_{\phi}(V|o_t, l)) \right]

   where :math:`D` is the dataset of trajectories, :math:`o_t` is the observation at timestep :math:`t`, :math:`l` is the language instruction, :math:`p_{\phi}(V|o_t, l)` is the predicted distribution over value bins, and :math:`H` is the cross-entropy loss.


Stage 3: Offline RL training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Sub-stage 1: Collect the dataset by rolling out the VLA policy on the single libero task
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The previously trained policy (t-1 th iteration policy for t th iteration and pretrained policy for first iteration) is used to rollout in libero simulation.
Roughly, 300 episodes are collected for the single libero task, which includes both success and failure episodes.

Sub-stage 2: Fine Tune the value function on collection of original dataset and all the previously rolled out dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The value function is fine-tuned on the collection of original dataset and all the previously rolled out dataset using the above mentioned procedure.
The value function is expected to prediction weighted sum of returns.
Pretained value function is used as a starting point for the fine-tuning and not the latest iteration value function as mentioned in the PI paper.


Sub-stage 3: Compute the advantage for each data point using the fine-tuned value function and calculate the epsilon threshold for setting I\ :sub:`t`\ (Indicator) VLA policy training.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The advantage is cmputed the classic RL advantage formula:
.. math::
    A(o_t, l) = V^B(o_N) + \sum_{t'=t}^{N} r_{t'} - V(o_t)
where :math:`V^B(o_N)` is the predicted value by the value function at N steps look ahead, :math:`r_{t'}` is the reward at timestep :math:`t'`, and :math:`V(o_t)` is the predicted value by the value function at timestep :math:`t`.

Continuous value function is calculated instead of discretized value function to get a more accurate advantage calculation. The continuous value function :math:`V(o_t, l)` is computed as the expected value of the discretized distribution:

.. math::

    V(o_t, l) = \sum_{b=1}^{B} p_{\phi}(V = b | o_t, l) \cdot v_b

where :math:`p_{\phi}(V = b | o_t, l)` is the predicted probability for bin :math:`b`, :math:`v_b` is the center value of bin :math:`b`, and :math:`B = 201` is the total number of bins.

The bins are evenly spaced between -1 and 0. So, mid value is used for each bin to compute the continuous value.


Once the advantage is calculated, the epsilon threshold is calculated such that 30% of the data points have positive advantage and remaining 70% have negative advantage.
Instead of using sample of dataset to calculate the epsilon threshold (as mentioned in the PI paper), we use whole dataset to calculate the epsilon threshold.

Sub-stage 4: Fine Tune the VLA policy on collection of original dataset and all the previously rolled out dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The VLA policy is fine-tuned on the collection of original dataset and all the previously rolled out dataset using the above mentioned procedure.
Pretained VLA policy is used as a starting point for the fine-tuning and not the latest iteration VLA policy as mentioned in the PI paper.
The I\ :sub:`t`\ (Indicator) VLA policy training is set to True (I\ :sub:`t`\ = "Advantage : Positive") for all the data points where the advantage is greater than the epsilon threshold.
