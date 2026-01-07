<p align="center">
  <a href="https://www.tensor.auto">
    <img src="assets/logo.png" alt="Logo">
  </a>
</p>

# OpenTau - Train VLA models with state-of-the-art techniques by Tensor

At Tensor, we are pushing the frontier of large foundation models for physical AI. In robot learning, a vision-language-action (VLA) model is a multimodal foundation model that integrates vision, language, and action. Today, VLA represents the leading approach for embodied AI, spanning autonomous driving, robot manipulation, and navigation.

OpenTau is Tensor’s open-source training toolchain for frontier VLA models—designed to make training reproducible, accessible, and scalable. At Tensor, we believe in open research and reproducible progress for the robotics community. By open-sourcing our training toolchain, we aim to expand knowledge sharing and accelerate scientific progress that others can reproduce.

Whether you use the official OpenPi codebase or LeRobot’s reimplementation, you may still be missing key components. OpenTau implements these key capabilities in one place:

- Co-training on an adjustable mixture of heterogeneous datasets
- Discrete actions for fast VLM convergence in $\pi_{0.5}$
- Knowledge insulation between the VLM backbone and the action expert
- Dropout in the VLM to reduce overfitting
- A reinforcement learning pipeline described in $\pi^*_{0.6}$
- And more...

OpenTau ($\tau$) is a tool developed by *[Tensor][1]* to bridge this gap, and we also use it internally to train our proprietary in-house models. Our goal is to help you train VLAs on any dataset while fully leveraging state-of-the-art techniques. We plan to continuously upgrade this repository to keep pace with the state of the art in the robotics community.

| Features                                                 | OpenPi                  | LeRobot                          | **OpenTau** |
| -------------------------------------------------------: | :---------------------: | :------------------------------: | :---------: |
| Co-training with Heterogeneous Datasets                  | ❌                       | ❌                                | ✅           |
| Discrete Actions Training in $\pi_{0.5}$                 | ❌                       | ❌                                | ✅           |
| Knowledge Insulation (KI) between VLM and Action Decoder | ❌                       | ❌                                | ✅           |
| Dropout Layers in PaliGemma                              | ✅ (Jax) <br>❌ (PyTorch) | ❌                                | ✅           |
| Multi-Node and Multi-GPU Training                        | ❌                       | ✅                                | ✅           |
| Fully Functioning $\pi_{0.5}$ Checkpoint                 | ✅                       | ❌ <br> (Missing Text Embeddings) | ✅           |
| Simulation Environments for Evaluating Models            | ❌                       | ✅                                | ✅           |
| $\pi^{*}_{0.6}$ style Reinforcement Learning Pipeline    | ❌                       | ❌                                | ✅           |
| Framework                                                | Jax / PyTorch           | PyTorch                          | PyTorch     |

## Quick Start
If you are familiar with LeRobot, getting started with OpenTau is very easy.
Because OpenTau is a fork of the popular LeRobot repository, any LeRobot-compliant policy and dataset can be used directly with OpenTau.
Check out our documentation to get started quickly.
We provide a quick start guide to help you get started with OpenTau.

For using local notebooks to train and evaluate models, find the notebooks at `notebooks/pi05_training.ipynb` and `notebooks/pi05_evaluation_only.ipynb`.

For using the Google Colab notebooks to train and evaluate models, find the colab notebooks here: [pi05_training](https://colab.research.google.com/drive/1DeU0lNnEzs1KHo0Nkgh4YKBr-xu9moBM?usp=sharing) and [pi05_evaluation_only](https://colab.research.google.com/drive/1U_AyuH9WYMT4anEWvsOtIT7g01jA0WGm?usp=sharing) respectively.

## Checkpoints
We provide fully functioning $\pi_{0.5}$ checkpoints trained with high success rates. We plan to release more models in the near future.

| Model Checkpoint              | Description                                                                                                   | Success Rate (%)                                                   |
|-------------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [TensorAuto/tPi0.5-libero][2] | A $\pi_{0.5}$ model checkpoint trained on the LIBERO dataset with discrete actions and knowledge insulation.  | 98.4% (10) <br> 97.6% (Goal) <br> 100% (Object) <br> 98% (Spatial) |
| [TensorAuto/pi05_base][5]     | A $\pi_{0.5}$ model checkpoint converted from the official openpi checkpoint, with language embeddings added. | N/A                                                                |
| More coming soon...           |                                                                                                               |                                                                    |

## Acknowledgements

This project builds on the $\pi$ series of [papers][3] and many other open-source efforts—especially [LeRobot][4]—for re-implementing the $\pi$ models and helping standardize training infrastructure. OpenTau extends these foundations to provide a more accessible, comprehensive toolchain for training vision-language-action agents.

[1]:	https://www.tensor.ai
[2]:	https://huggingface.co/TensorAuto/tPi0.5-libero
[3]:	https://www.pi.website/blog
[4]:	https://huggingface.co/lerobot
[5]:    https://huggingface.co/TensorAuto/pi05_base
