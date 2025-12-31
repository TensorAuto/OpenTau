This tutorial demonstrates how we train a pi0 policy on libero dataset using offline RL that closely follows the training procedure of OpenPI on pi06-star.

The procedure is as follows:

1. Pre-Train VLA policy on whole libero dataset till convergence.
2. Pre-Train value function on whole libero dataset till convergence.
3. Fine Tune the VLA policy on single libero task till convergence.
4. Repeat the below steps for 1-3 times:
    i. Collect the dataset by rolling out the VLA policy on the single libero task
    ii. Fine Tune the value function on collection of original dataset and all the previously rolled out dataset
    iii. Compute the advantage for each data point using the fine-tuned value function and calculate the epsilon threshold for setting I\ :sub:`t`\ (Indicator) VLA policy training.
    iii. Fine Tune the VLA policy on collection of original dataset and all the previously rolled out dataset
