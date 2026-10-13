# CS3268 Responsible AI: From Algorithms to Impact<br>Project Document

**Assessment Weight:** 25% of total grade

- Proposal 5%
- Group Assessment (Presentation + Report) 10%
- Individual Assessment (Presentation) 10 %

**Team Size:** 4

**Deliverables:**

1. Proposal Document
2. Report Document
3. Presentation Slides
4. Code Assets

## 1 Context

The project offers a way to expand your knowledge horizon beyond what the lectures cover. Use this as an opportunity to increase the depth and breadth of your knowledge of Responsible AI. You will do that by identifying real-world risks, understanding technical methods and their trade-offs, and developing strategies for safe and reliable AI systems.

## 2 Project Tracks

Pick one of the three tracks for your project:

1. Implementation
2. Evaluation
3. Survey

### 2.1 Track 1: Implementation

In this track, your team will build an end-to-end Machine Learning system for a realistic use case. Rather than optimizing solely for raw predictive performance (e.g., Accuracy, F1-score), your goal is to embed at least two Responsible AI (RAI) principles into the ML lifecycle and analyze how these methods interact with each other and impact overall performance.

1. Train a standard, functioning ML baseline system.
2. Think of potential technical or societal risks present in your baseline (e.g., bias, privacy leaks, adversarial vulnerability).
3. Implement methods targeting at least two distinct RAI dimensions (e.g., Fairness + Interpretability, or Privacy + Robustness).
4. Evaluate how the chosen RAI interventions affect model performance, how they impact one another and any trade-offs.

#### 2.1.1 Example Domains

This is a non-exhaustive list, you may come up with your own topics:

- Healthcare Diagnosis: Training private (DP) and interpretable diagnostic classifiers.
- Toxic Content Moderation: Building robust and transparent moderation tools for university chatbots.
- Pedestrian Detection: Testing object detection models for demographic fairness and environmental robustness.
- E-Commerce Recommendation: Balancing popularity bias with user privacy.

### 2.2 Track 2: Evaluation

In this track, your team will conduct an empirical benchmarking study comparing existing research methods that address a specific dimension, such as privacy, fairness, robustness, or interpretability/explainability. Rather than inventing a new algorithm, your goal is to reimplement and standardize these methods under identical experimental conditions (same dataset, baseline architecture, and metrics) to critically analyze their strengths, weaknesses, and edge cases.

1. Pick two or more published research papers or established algorithms that address a particular Responsible AI challenge.
2. Standardize the benchmark by training both methods on the same dataset(s), baseline models, and evaluation metrics.
3. Set up and run the codebases under identical conditions to ensure a fair head-to-head comparison.
4. Evaluate trade-offs and edge cases beyond basic accuracy.

### 2.3 Track 3: Survey

In this track, your team will conduct a deep, collaborative literature review on a focused open problem within Responsible AI. Rather than each team member writing isolated paper summaries, your goal is to stitch together findings across a wide range of literature - find recurring patterns, build taxonomies, and highlight open gaps yet to be tackled.

1. Survey a substantial body of literature (12+ papers).
2. Work together to build shared artifacts — such as a taxonomy of approaches, feature comparison matrices, plots etc.
3. Identify common limitations, conflicting findings across papers, trade-offs between methods, and edge cases.
4. Consolidate your findings to propose future research directions and open questions.

#### 2.3.1 Example Survey Topics

This is a non-exhaustive list, you may come up with your own topics:

- Jailbreaking & Safety Defenses in LLMs
- Watermarking
- Deepfake detection
- Hallucination Mitigation
- Mechanistic Interpretability
- Counterfactual Explanations
- Federated Learning
- Machine Unlearning

## 3 Proposal

- Submission Deadline: <span style="color: #ff0000;">8 September 2026, 10:00 AM</span> via Canvas.
- Deliverable: Proposal Document. Maximum 2 pages, single-column format. The 2-page limit includes all references.
- Your project group MUST be registered on Canvas before submitting the proposal. This ensures clean submissions for your team.
  - Group Registration is on Canvas, closes 8 September 2026, 10:00 AM.
- Propose a title for your project.
- Clearly display all team members’ Full Names, Student Numbers (Axxxxxxx) and NUSNET IDs (exxxxxx) at the top of the first page.
- Your 2-page proposal must explicitly state your chosen track and address the specific items outlined below:
  - Track 1:
    - Define the target ML problem and the identify dataset you plan to use.
    - Describe the standard machine learning model you will build as your starting baseline.
    - Identify at least two Responsible AI principles to integrate and its respective proposed methods.
  - Track 2:
    - List 2 or more published research papers/ methods targeting the focus area that you will evaluate.
    - Describe the standardized environment (dataset, baseline architecture, and parameters) you will use for a direct head-to-head benchmark.
    - Detail the metric(s) you will use to evaluate the methods.
  - Track 3:
    - Define the focused domain in Responsible AI your team will review.
    - Outline the key research questions your survey will answer.
    - Provide an initial list of key research papers that show the evidence of literature.

## 4 Final Presentation

- Format: In-person Face-to-Face (F2F) presentation.
- Date: 10 November 2026 during lecture and tutorial times. Depending on the total number of registered teams in the course, additional presentation slots may be allocated.
- Submission Deadline: <span style="color: #ff0000;">10 November 2026, 10:00 AM</span> via Canvas.
- Deliverable: Presentation Slide Deck.

Further details will be communicated closer to the presentation date.

## 5 Report

- Submission Deadline: <span style="color: #ff0000;">10 November 2026, 10:00 AM</span> via Canvas.
- Document Format: Use the template that will be shared with you on Canvas.
- Page Limit: Maximum 6 pages for main content + 1 page strictly reserved for references (7 pages total).
- Your final report must follow standard academic writing guidelines and include the core sections:
  1. Abstract
  2. Introduction
  3. Related Work
  4. Method
  5. Experiments/ Results
  6. Conclusion & Future Work

## 6 Appendix: Resources

Here are some resources to get you started:

### 6.1 Fairness

| Tool | Main Capabilities |
| :---: | :---: |
| [Fairlearn](https://fairlearn.org/) | Group-wise assessment, metric frames, reductions, post-processing |
| [AI Fairness 360 (AIF360)](https://github.com/Trusted-AI/AIF360) | Broad collection of fairness metrics and mitigation algorithms |

### 6.2 Differential Privacy & Federated Learning

| Tool | Main Capabilities |
| :---: | :---: |
| [Opacus](https://opacus.ai/) | Differentially private training for PyTorch with privacy accounting |
| [Flower AI](https://flower.ai/) | Framework-agnostic federated learning and decentralized model training |
| [diffprivlib](https://diffprivlib.readthedocs.io/) | Differentially private statistics and classical ML algorithms |
| [TensorFlow Privacy](https://www.tensorflow.org/responsible_ai/privacy/guide) | DP training and privacy-analysis tools in TensorFlow |

### 6.3 Robustness & Security

| Tool | Main Capabilities |
| :---: | :---: |
| [Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.readthedocs.io/) | Evasion, poisoning, extraction, inference, defences, and metrics |
| [RobustBench](https://robustbench.github.io/) | Standardised adversarial-robustness models and evaluation |
| [TextAttack](https://textattack.readthedocs.io/) | Adversarial attacks, augmentation, and evaluation for NLP models |
| [WILDS](https://wilds.stanford.edu/) | Datasets for distribution shifts |

### 6.4 Interpretability

| Tool | Main Capabilities |
| :---: | :---: |
| [SHAP](https://shap.readthedocs.io/) | Game-theoretic feature attribution across model classes |
| [InterpretML](https://interpret.ml/) | Glass-box models (EBMs) and black-box explanation methods |
| [Captum](https://captum.ai/) | PyTorch attribution and interpretability algorithms |
| [TransformerLens](https://github.com/neelnanda-io/TransformerLens) | Mechanistic interpretability for autoregressive language models |

### 6.5 GenAI Evaluation

| Tool | Main Capabilities |
| :---: | :---: |
| [Inspect AI](https://inspect.aisi.org.uk/) | Composable LLM evaluation tasks, datasets, solvers, and scorers |
| [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) | Standardised jailbreak attack evaluation protocol and benchmarks |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | Automated red-team evaluation framework and classifiers |
| [PEFT](https://huggingface.co/docs/peft/) | LoRA, QLoRA, and parameter-efficient fine-tuning methods |
