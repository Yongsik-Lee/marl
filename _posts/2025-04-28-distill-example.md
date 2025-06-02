---
layout: distill
title: AI810 Blog Post (20205266)
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text. 
  Do not include math/latex or hyperlinks.
date: 2025-04-28
future: true
htmlwidgets: true
#hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Yongsik Lee
    affiliations:
      name: KAIST

# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Regulatory DNA Sequence Design with Reinforcement Learning  
    subsections:
    - name: Overview
    - name: Method
    - name: Results
    - name: Personal Take & Commentary

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

# Regulatory DNA Sequence Design with Reinforcement Learning  

## Overview

### Motivation

The design of **cis-regulatory elements (CREs)**—short DNA sequences like enhancers and promoters that control gene expression—is foundational for progress in synthetic biology, gene therapy, and precision medicine. While millions of CREs have been identified, natural evolution only explores a sparse region of the vast sequence space, and traditional methods like directed evolution or simple search-based optimizations are either slow, expensive, or prone to local optima.

### Problem Statement

Designing high-fitness CREs involves finding DNA sequences that induce desired gene expression levels. Existing computational approaches often:
1. Rely on local search around existing sequences, suffering from poor diversity and local optima.
2. Ignore biological priors, such as the known effects of transcription factor binding sites (TFBSs).

### Key Contributions

This paper introduces **TACO (TFBS-Aware Cis-regulatory element Optimization)**, a **reinforcement learning (RL)**-based method that fine-tunes a pre-trained autoregressive (AR) DNA model (HyenaDNA) to generate high-fitness, diverse CREs. The method incorporates **biological priors** by shaping rewards using TFBS information inferred via SHAP analysis on a surrogate LightGBM model. The contributions are:

- A novel RL fine-tuning framework for DNA sequence generation.
- Integration of TFBS knowledge to inform and steer sequence optimization.
- Comprehensive experiments across yeast and human datasets demonstrating improved performance and diversity.



## Method

### Core Model Architecture or Theoretical Framework

The authors frame sequence generation as a **Markov Decision Process (MDP)**:
- **States**: Partially generated DNA sequences.
- **Actions**: Next nucleotide (A, T, C, G).
- **Policy**: The AR model $\pi_\theta$, initialized with HyenaDNA, fine-tuned via RL.
- **Rewards**:

$$
r(s_{i-1}, a_i) = 
\begin{cases}
r_{\text{fitness}}, & \text{if } i = L \\
r_{\text{TFBS}}(t), & \text{if } a_i \text{ forms TFBS } t \\
0, & \text{otherwise}
\end{cases}
$$

where $r_{\text{TFBS}}(t) = \alpha \cdot \mu_{\phi}(t)$ if $p\text{-value} < 0.05$, using mean SHAP values $\mu_{\phi}(t)$.

Optimization uses **REINFORCE** with entropy regularization and a **hill-climbing replay buffer**, enhancing exploration and guiding the agent toward biologically meaningful regions of sequence space.



## Results

### Experimental Summary

The evaluation covers two domains:
- **Yeast promoters** (short sequences, simpler regulatory grammar).
- **Human enhancers** (longer, cell-type-specific complexity).

Settings:
- **Active Learning**: The oracle is visible during training.
- **Offline Model-Based Optimization (MBO)**: The oracle is hidden; optimization relies on a surrogate.

Baselines include BO, CMAES, AdaLead, PEX, and DNARL.

### Empirical Analysis

- **Yeast Results**: All models achieved max fitness, but **TACO surpassed all in diversity**, suggesting better exploration.
- **Human Enhancers**:
  - **TACO maintained the highest diversity** while achieving comparable or superior fitness, particularly in the K562 and SK-N-SH cell lines.
  - **PEX** had high fitness but low diversity.
  - **AdaLead** was fast but prone to premature convergence.
- **Offline MBO**: TACO preserved its edge in diversity and achieved strong performance without oracle access.

Notably, TACO's use of **biologically informed rewards** enabled it to escape local optima and avoid overfitting to surrogate models, a known failure mode in offline optimization.

### Strengths

- **Innovative use of TFBS-informed rewards**
- **High diversity and fitness**—critical in real applications
- **Robust performance** across domains and training settings
- **Strong theoretical grounding** in MDP and interpretable feature attribution

### Limitations

- **Static TFBS vocabulary**: Based on a fixed database; dynamic motif discovery is unexplored.
- **No modeling of TF interactions or orientation**: Limits biological realism.
- **Fitness oracle is assumed accurate**: Real-world biological validation is absent.
- **Generality across organisms**: Not yet demonstrated outside yeast/human.



## Personal Take & Commentary

### Interpretation from My Research Background

As a researcher in **RL and multi-agent systems**, particularly with recent work in **LLM-based agents**, this paper feels like a well-integrated application of RL to structured sequence design. The use of a generative AR model as a policy echoes **autoregressive planning** in agents, and the reward shaping via TFBS knowledge mirrors **value shaping in multi-agent learning** to incorporate domain priors.

Moreover, the method’s ability to balance **exploration (diversity)** and **exploitation (fitness)** under surrogate uncertainty aligns with ideas from **conservative RL** and **offline policy optimization** in LLM agents. The idea of inferring "reward models" from interpretable features also shares philosophical ground with reward learning in autonomous agents.

### Implications for Future Research

- **Cross-domain generalization**: Could TACO-style methods transfer to protein engineering or RNA design with appropriate priors?
- **Multi-objective optimization**: Incorporating trade-offs (e.g., tissue specificity vs. stability) could benefit from **Pareto front-based RL**.
- **Hierarchical action spaces**: DNA motifs are naturally hierarchical. Could **options or temporally extended actions** improve sample efficiency?
- **Language model parallels**: Applying chain-of-thought prompting or instruction tuning to DNA generation is an intriguing idea.

### Critical Commentary

- While TACO clearly advances the state of the art, it remains dependent on oracle quality. Exploring **uncertainty-aware RL** or **model-based planning** using ensembles could mitigate this.
- The **lack of experimental wet-lab validation** leaves the practical impact of the method speculative, though this is a systemic issue in computational biology.
- The method could benefit from **probabilistic modeling of biological constraints**, perhaps by fusing the AR model with a **variational framework**.



## 🌟 TL;DR

**TACO is a biologically-aware RL method for DNA sequence design that integrates generative modeling and interpretable reward shaping.** It sets a new benchmark in CRE optimization by balancing fitness and diversity—two goals often in conflict—and represents a compelling bridge between computational biology and reinforcement learning. For RL researchers, this is a refreshing demonstration of classic ideas in a novel and high-impact domain.