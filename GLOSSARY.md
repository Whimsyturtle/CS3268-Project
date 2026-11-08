# Glossary

Preferred terms used in this repository. See the [proposal](CS3268_Project_Proposal_Group_6.md) for experiment design, model choices, and benchmark metrics.

## Activation

An internal numerical representation computed by a model while processing an input.

## Activation steering

Changing a model's internal activations during inference to influence its behavior while keeping the base model's parameters fixed.

## AlpacaEval 2.0

A benchmark for open-ended instruction following that uses an LLM as a judge of response quality.

## BBQ

A benchmark for measuring stereotype reliance in multiple-choice question answering.

## AUROC

Short for area under the receiver operating characteristic curve. It measures how
well scores rank positive examples above negative ones, with ties counting half;
0.5 indicates chance-level ranking.

## CAA

Short for Contrastive Activation Addition, a method that constructs a steering vector from the mean difference between activations for contrasting examples.

## Bootstrap confidence interval

An estimate of uncertainty obtained by repeatedly sampling the observed examples
with replacement and recomputing a statistic. Paired comparisons resample each
prompt's baseline and steered result together.

## Checkpoint

A saved set of trained model parameters, loaded to reuse a model without training
it again.

## Compute cluster

A group of connected computers used to run computing work. See the [cluster guide](docs/SoC_Cluster_Guide.md) for access and setup.

## CUDA

NVIDIA's platform for running computations on its GPUs. PyTorch's `cu126` build
label means CUDA 12.6; it does not identify the installed NVIDIA driver version.

## Decoding hyperparameters

Settings that control how a model generates text.

## Decoder direction

The vector that an SAE uses to map one feature's activation back into the model's
activation space.

## Gemma 3

A family of language models; see the [proposal](CS3268_Project_Proposal_Group_6.md) for the model sizes planned for this study.

## General capability

A model's ability to perform tasks beyond the targeted steering behavior, such as reasoning and instruction following.

## Greedy decoding

Generating text by always choosing the most likely next token, so outputs do not depend on random sampling.

## Held-out evaluation prompts

Prompts reserved for evaluating model behavior, separate from the data used to construct steering vectors.

## HolisticBias

A benchmark for measuring demographic bias in open-ended text generation.

## Full Gen Bias

A HolisticBias metric that measures how much the complete distribution of predicted
conversational styles varies across demographic descriptors.

## Partial Gen Bias

The contribution of the individual styles in one style cluster to Full Gen Bias.

## Summed-Cluster Gen Bias

A HolisticBias metric that first combines probabilities for related conversational
styles, then measures how that combined probability varies across descriptors.

## ICV

Short for In-Context Vector, a method that extracts a reusable activation vector from demonstration pairs.

## Inference

Running a model on an input to produce an output.

## Intervention

The vectors, SAE feature selections, or learned ReFT-r1 parameters applied to steer the model. See the [directory structure](README.md#directory-structure) for storage locations.

## Issue

A GitHub record for a task's outcome, dependencies, progress, and completion checks. See the [contribution workflow](README.md#contributing).

## LLM

Short for Large Language Model, a model trained on large amounts of text to process and generate language.

## Milestone

A GitHub grouping of issues for a delivery stage.

## MIG

Short for Multi-Instance GPU, NVIDIA's division of a physical GPU into isolated
partitions with assigned memory and compute resources.

## MMLU-Pro

A benchmark for measuring general capability through multiple-choice reasoning questions.

## Parameter

A numerical value in a model. A **trainable parameter** is one that the training process can adjust.

## ReFT-r1

Short for Rank-1 Representation Finetuning, a method that learns a single hidden-state direction from supervised examples.

## Runtime

The software and hardware used to execute a run. The
[cluster requirements](docs/Cluster_Runtime.md)
define the shared setup and the evidence retained for new cluster runs.

## SAE

Short for Sparse Autoencoder, a model that reconstructs activations using a small number of active features.

## SAE feature

One component of an SAE's sparse representation, with an activation value and a
decoder direction.

## Residual stream

The internal representation passed between transformer layers and updated by
their attention and feed-forward computations.

## SM

Short for streaming multiprocessor, a GPU compute unit. A MIG partition can expose
fewer SMs than the full GPU.

## Social bias

Unfair patterns in a model's responses about social or demographic groups, such as reliance on stereotypes.

## Steering vector

A direction in a model's activation space used to influence its behavior.

## Unsteered baseline

The base model evaluated without activation steering or changes to the evaluation prompts.
