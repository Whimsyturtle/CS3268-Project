# Track 2: Comparing LLM Steering Methods for Bias Mitigation

## 1 Introduction

Large Language Models (LLMs) can reproduce societal biases present in their training data (Hu et al., 2025). Common mitigation approaches such as dataset filtering, SFT, and RLHF modify model weights and may affect existing capabilities. Activation steering offers an inference-time alternative by modifying internal representations without updating the base model. In this work, we compare four activation-steering methods on their ability to reduce social bias while preserving general model capability.

## 2 Literature Review

**Contrastive Activation Addition (CAA)** was proposed by Rimsky et al. (2024) as a simple, computationally efficient and reversible method to control socially undesirable behaviors such as bias while minimizing changes to the model’s underlying knowledge. It is an inference-time activation-steering method that modifies a LM’s internal representations without updating its parameters. CAA constructs a steering vector by averaging the difference between hidden-state activations generated from contrasting examples, such as responses exhibiting biased and unbiased behavior. This vector captures a behavioral direction associated with the desired change and is added to the model’s activations during generation.

**In-Context Vector (ICV)**, proposed by Liu et al. (2023), was inspired by in-context learning (ICL), where the model learns to produce desired behaviours based on the demonstration examples provided. However, unlike ICL where the examples are directly injected into each input prompt, ICV extracts this shift in latent states caused by the examples into a reusable activation vector that can be added to various layers in the LLM. For a given layer, we first compute the difference between intermediate activation states of the desired output and input for each example pair. PCA is then performed on the stacked difference vectors, and the first principal component is used as the ICV for that layer.

**Sparse Autoencoder (SAE)** usage for interpreting a LM’s internal representations was proposed by Cunningham et al. (2023). Given an internal activation vector from a selected layer, an SAE learns to approximately reconstruct it using only a small number of active features. This yields more interpretable feature directions, from which the directions associated with the target behavior can be selected and used to steer the LLM.

**ReFT-r1 (Rank-1 Representation Finetuning)** was introduced by Wu et al. (2025) in AxBench. It learns a single hidden-state direction from supervised examples and uses it for both concept detection and generation steering.

## 3 Methodology

We aim to use the model Gemma-3-1b for our experiments. Depending on available compute, we may extend our experiments to Gemma-3-4b as well. We chose Gemma 3 due to its high performance compared to other open source models of a similar size, and the availability of pre-trained sparse autoencoders from GemmaScope.

### 3.1 Overview of Steering Methods

| Method | Steering Signal | How Steering is Achieved | Internals Used |
| --- | --- | --- | --- |
| **CAA** | Dense vector | Mean activation difference | Hidden states |
| **ICV** | Dense vector | Latent shift from demonstrations | Hidden states |
| **SAE** | Sparse feature | Select pretrained SAE feature | Hidden states + SAE |
| **ReFT-r1** | Learned vector | Optimized on supervised examples | Hidden states + gradients |

### 3.2 Dataset and Metrics

| Benchmark | Task | Measures | Metric(s) | Purpose |
| --- | --- | --- | --- | --- |
| BBQ | MCQ QA | Stereotype reliance | Bias score<br>Accuracy | Bias in ambiguated / disambiguated contexts |
| HolisticBias | Open-ended text generation | Demographic bias | Full Gen Bias<br>Summed-Cluster Gen Bias | Free-form generalization |
| MMLU-Pro | MCQ reasoning | General capability | Accuracy | Ensures that text generation quality does not degrade significantly |
| AlpacaEval 2.0 | Open-ended instruction following | General response quality | LLM-as-a-judge | Ensures steering does not degrade normal generation quality |

All methods use identical held-out prompts and LLM decoding hyperparameters. Here, our baseline that directly queries the unsteered LLM with unmodified evaluation prompts serves as the control to evaluate the effect of each steering method. Note that data used to construct the activation vectors are disjoint from our evaluation datasets.

## References

Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600.

Hu, T., Kyrychenko, Y., Rathje, S., Collier, N., Van Der Linden, S., & Roozenbeek, J. (2025). Generative language models exhibit social identity biases. Nature Computational Science, 5(1), 65–75.

Liu, S., Ye, H., Xing, L., & Zou, J. Y. (2024). In-context vectors: Making in-context learning more effective and controllable through latent space steering. Proceedings of the 41st International Conference on Machine Learning, 235, 32287–32307.

Parrish, A., Chen, A., Nangia, N., Padmakumar, V., Phang, J., Thompson, J., Htut, P. M., & Bowman, S. R. (2022). BBQ: A hand-built bias benchmark for question answering. Findings of the Association for Computational Linguistics: ACL 2022, 2086–2105.

Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. (2024). Steering Llama 2 via contrastive activation addition. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15504–15522.

Smith, E. M., Hall, M., Kambadur, M., Presani, E., & Williams, A. (2022). “I’m sorry to hear that”: Finding new biases in language models with a holistic descriptor dataset. Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 9180–9211.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., & Chen, W. (2024). MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37.

Wu, Z., Arora, A., Geiger, A., Wang, Z., Huang, J., Jurafsky, D., Manning, C. D., & Potts, C. (2025). AxBench: Steering LLMs? Even simple baselines outperform sparse autoencoders. Proceedings of the 42nd International Conference on Machine Learning.
