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

## CAA

Short for Contrastive Activation Addition, a method that constructs a steering vector from the mean difference between activations for contrasting examples.

## Compute cluster

A group of connected computers used to run computing work. See the [cluster guide](docs/SoC_Cluster_Guide.md) for access and setup.

## Decoding hyperparameters

Settings that control how a model generates text.

## Gemma 3

A family of language models; see the [proposal](CS3268_Project_Proposal_Group_6.md) for the model sizes planned for this study.

## General capability

A model's ability to perform tasks beyond the targeted steering behavior, such as reasoning and instruction following.

## Held-out evaluation prompts

Prompts reserved for evaluating model behavior, separate from the data used to construct steering vectors.

## HolisticBias

A benchmark for measuring demographic bias in open-ended text generation.

## ICV

Short for In-Context Vector, a method that extracts a reusable activation vector from demonstration pairs.

## Inference

Running a model on an input to produce an output.

## Intervention

The vectors, SAE feature selections, or learned ReFT-r1 parameters applied to steer the model. Saved interventions and their loading settings are stored under `checkpoints/`.

## LLM

Short for Large Language Model, a model trained on large amounts of text to process and generate language.

## MMLU-Pro

A benchmark for measuring general capability through multiple-choice reasoning questions.

## Parameter

A numerical value in a model. A **trainable parameter** is one that the training process can adjust.

## ReFT-r1

Short for Rank-1 Representation Finetuning, a method that learns a single hidden-state direction from supervised examples.

## SAE

Short for Sparse Autoencoder, a model that reconstructs activations using a small number of active features.

## Social bias

Unfair patterns in a model's responses about social or demographic groups, such as reliance on stereotypes.

## Steering vector

A direction in a model's activation space used to influence its behavior.

## Unsteered baseline

The base model evaluated without activation steering or changes to the evaluation prompts.
