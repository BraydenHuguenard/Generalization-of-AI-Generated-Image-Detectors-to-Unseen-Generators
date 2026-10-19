# Generalization of AI-Generated Image Detectors to Unseen Generators

## Overview

As image generation models continue to improve, distinguishing real images from AI-generated images has become increasingly challenging. Although AI-generated image detectors can achieve strong performance on images from generators represented in their training data, they may struggle when exposed to previously unseen generators.

This project investigates **how well AI-generated image detectors generalize to unseen image generators** and whether detector architecture influences generalization. We will fine-tune and compare two convolutional neural networks (CNNs) and a Vision Transformer (ViT) on real and AI-generated images. By withholding different generators during training, we aim to identify generalization gaps, architecture-dependent failure patterns, and potential reliance on generator-specific artifacts.

> **Status:** Research and experimentation in progress. Results will be added as experiments are completed.

## Research Questions

1. Can AI-generated image detectors trained on a subset of generators accurately identify images produced by unseen generators?
2. To what extent do detectors learn generalizable features versus generator-specific artifacts or dataset shortcuts?
3. How does detector architecture affect generalization to unseen generators?
4. Do CNN-based and Transformer-based detectors exhibit different failure patterns?

## Models

We plan to fine-tune three pretrained image classification architectures for binary classification:

| Architecture | Family | Role |
| --- | --- | --- |
| ResNet-50 | CNN | Convolutional baseline |
| ConvNeXt-Tiny | CNN | Modern convolutional baseline |
| ViT-Small/16 | Vision Transformer | Transformer-based comparison |

**Input:** RGB image, resized and normalized consistently across models.  
**Output:** Probability that the image is AI-generated.  
**Training objective:** Binary cross-entropy loss.

## Datasets

We plan to construct balanced experimental subsets from the following public datasets:

| Dataset | Approximate size | Description |
| --- | --- | --- |
| [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) | 42K images | Real and AI-generated images spanning five generator models |
| [Community Forensics](https://arxiv.org/abs/2411.04125) | 2.7M images | Broad collection associated with thousands of image generators |

We may only use the Defactify dataset if we find the data adequate. The full datasets may be too large for repeated experiments, so we will sample representative subsets while attempting to balance real and generated examples and control for differences in content and preprocessing.

## Experimental Methodology

### 1. Data preparation

- Select representative generator subsets and real-image examples.
- Standardize image resizing and normalization.
- Use separate training, validation, and test partitions, taking care to avoid duplicate or near-duplicate images across splits.
- Record dataset and generator provenance so that performance can be attributed to the correct sources.

### 2. Model fine-tuning

- Initialize ResNet-50, ConvNeXt-Tiny, and ViT-Small/16 with pretrained weights.
- Adapt each model to output a binary classification score.
- Fine-tune under a comparable training and evaluation procedure using binary cross-entropy loss.

### 3. Held-out-generator evaluation

- Train detectors using images from a subset of generators.
- Evaluate on **seen generators** (represented during training) and **unseen generators** (excluded from training).
- Repeat experiments while withholding different generators to assess sensitivity to generator identity.
- Compare performance across architectures and individual generators.

### 4. Failure analysis

Where feasible, examine misclassified images and recurring patterns. Potential follow-up analyses include assessing whether detectors rely on resolution, content, or generator-specific visual artifacts, as well as testing sensitivity to common image transformations.

## Evaluation

We will report the following classification metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**

Our primary comparison will be the **generalization gap** between seen- and unseen-generator performance. For a metric where higher is better, this can be expressed as:

\[
\text{Generalization Gap} = \text{Score}_{\text{seen}} - \text{Score}_{\text{unseen}}
\]

We aim to present per-generator results, architecture comparisons, and visualizations of failure cases. Results are not yet available.

## Compute Environment

Experiments are planned using:

- **Georgia Tech PACE** high-performance computing resources
- **Google Colab** for prototyping and supplementary experiments

Dataset size, storage requirements, and training time will influence sampling and experiment scope.

## Challenges and Limitations

- **Data scale:** Large datasets increase storage, preprocessing, and training costs.
- **Dataset shortcuts:** Models may exploit image content, resolution, compression, or source differences instead of artifacts specific to generated imagery.
- **Unseen-generator shift:** Performance on current held-out generators may not predict performance on future image generation models.
- **Controlled comparisons:** Architecture conclusions require attention to preprocessing, pretrained weights, training budgets, and dataset composition.

## References

1. Fabrizio, G., et al. (2024). *A Bias-Free Training Paradigm for More General AI-generated Image Detection*. [arXiv:2412.17671](https://arxiv.org/abs/2412.17671).
2. Imanpour, N., et al. (2024). *The Visual Counter Turing Test (VCT2): A Benchmark for Evaluating AI-Generated Image Detection and the Visual AI Index (VAI)*. [arXiv:2411.16754](https://arxiv.org/abs/2411.16754).
3. Karageorgiou, D., et al. (2024). *Any-Resolution AI-Generated Image Detection by Spectral Learning*. [arXiv:2411.19417](https://arxiv.org/abs/2411.19417).
4. Park, J., & Owens, A. (2024). *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*. [arXiv:2411.04125](https://arxiv.org/abs/2411.04125).
5. Roy, R., et al. (2026). *A Comprehensive Dataset for Human vs. AI Generated Image Detection*. [arXiv:2601.00553](https://arxiv.org/abs/2601.00553).
6. Wu, S., et al. (2025). *Few-Shot Learner Generalizes Across AI-Generated Image Detection*. [arXiv:2501.08763](https://arxiv.org/abs/2501.08763).
7. Yan, H., et al. (2024). *WildFake: A Large-scale Challenging Dataset for AI-Generated Images Detection*. [arXiv:2402.11843](https://arxiv.org/abs/2402.11843).
8. Yan, S., et al. (2024). *A Sanity Check for AI-generated Image Detection*. [arXiv:2406.19435](https://arxiv.org/abs/2406.19435).
9. Zhu, M., et al. (2023). *GenDet: Towards Good Generalizations for AI-Generated Image Detection*. Reference identifier as provided in project proposal; bibliographic link pending verification.

## Project Status

This repository accompanies an ongoing deep learning research project. Implementation details, reproducibility instructions, and experimental results will be documented here as they become available.
