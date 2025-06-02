---
layout: distill
title: AI810 Blog Post (20205266)
description: This blogpost reviews two ICLR 2025 papers: one on RL-guided DNA sequence design (TACO), and another on a unified generative modeling framework based on Markov processes (Generator Matching). Though from different domains, both highlight how structured priors and guided dynamics improve generative outcomes. Their pairing reveals shared insights relevant to reinforcement learning and LLM-based agents, especially in offline optimization, reward shaping, and multi-agent compositionality. For researchers in intelligent agent design, these works offer complementary views on controllable generation—whether in biology, vision, or language.
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
  - name: Introduction
  - name: Regulatory DNA Sequence Design with Reinforcement Learning  
    subsections:
    - name: Overview
    - name: Method
    - name: Results
    - name: Personal Take & Commentary
  - name: Generator Matching Generative Modeling with Arbitrary Markov Processes
    subsections:
    - name: Overview
    - name: Method
    - name: Results
    - name: Personal Take & Commentary
  - name: Conclusion

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


# Introduction

check

This review pairs two thematically distinct yet intellectually adjacent papers: “Regulatory DNA Sequence Design with Reinforcement Learning” and “Generator Matching: Generative Modeling with Arbitrary Markov Processes.” While one operates in the domain of biological sequence design and the other in foundational generative modeling, both tackle a common question at the heart of intelligent system design: how can we guide generative processes using structured priors and dynamic feedback?

The first paper explores how reinforcement learning—augmented with biological knowledge—can improve the optimization of cis-regulatory DNA sequences, a crucial problem in synthetic biology. The second paper introduces a unifying mathematical framework for generative modeling through the lens of parameterized Markov processes, offering a flexible alternative to existing paradigms like diffusion and flow models.

From the vantage point of RL and LLM-based agents, both works offer rich insights. TACO (Paper 1) exemplifies reward shaping and policy optimization in a constrained generative environment, while Generator Matching (Paper 2) provides a theoretical scaffold for modeling and composing stochastic transitions—akin to environment dynamics in model-based RL or planning behaviors in complex multi-agent systems. Despite addressing different domains, both papers converge on a shared ambition: to systematically construct or adapt generative mechanisms toward target outcomes using theoretically grounded tools.


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





## Generator Matching: Generative Modeling with Arbitrary Markov Processes

## Overview

### Motivation

In recent years, generative modeling has been dominated by paradigms such as **VAEs**, **GANs**, **Diffusion Models**, and **Flow Models**. Despite their differences, these models share a structural similarity: they operate over **Markovian transformations of probability distributions**, incrementally transforming simple priors into complex data distributions.

This paper introduces **Generator Matching (GM)**, a framework that unifies these approaches under the abstraction of **parameterized Markov generators**. This unification not only connects flow-based and diffusion models but also reveals previously unexplored classes of generative processes, such as jump processes, and provides a principled method for combining them.

### Problem Statement

How can we construct a scalable, flexible, and modality-agnostic generative modeling framework grounded in the theory of **Markov processes** that both unifies existing methods and enables the development of new ones?

### Key Contributions

- **Unified Framework**: Proposes **Generator Matching**, generalizing flow matching, diffusion models, and discrete diffusion models.
- **Universal Characterization**: Provides a comprehensive characterization of Markov process generators in Euclidean and discrete spaces.
- **New Model Class**: Introduces **jump models** in $\mathbb{R}^d$ as a novel generative modeling approach.
- **Model Combinations**: Introduces **Markov superpositions** and principled multimodal model composition.
- **Empirical Validation**: Demonstrates competitive performance in image and multimodal protein generation.


## Method

### Core Model Architecture / Theoretical Framework

The GM framework is based on the **infinitesimal generator** $L_t$ of a Markov process $(X_t)_{t \in [0,1]}$, satisfying the **Kolmogorov Forward Equation (KFE)**:

$$
\partial_t \mathbb{E}_{x \sim p_t}[f(x)] = \mathbb{E}_{x \sim p_t}[L_t f(x)]
$$

The approach involves:

- Selecting a **conditional probability path** $p_t(\cdot | z)$ that interpolates between a simple prior $p_0$ and a delta distribution at data point $z$.
- Deriving a **conditional generator** $L^z_t$ satisfying the KFE for $p_t(\cdot | z)$.
- Aggregating these to obtain the **marginal generator**:

$$
L_t f(x) = \mathbb{E}_{z \sim p_{1|t}(\cdot | x)}[L^z_t f(x)]
$$

Training involves minimizing a **conditional Generator Matching (CGM) loss** using a **Bregman divergence** between the ground truth and parameterized generator outputs.

In Euclidean space, the generator can be expressed as:

$$
L_t f(x) = \nabla f(x)^T u_t(x) + \frac{1}{2} \text{Tr}\left[\nabla^2 f(x) \cdot \sigma_t^2(x)\right] + \int [f(y) - f(x)] Q_t(dy; x)
$$

This formulation unifies:

- Flows: via drift $u_t$
- Diffusions: via noise $\sigma_t$
- Jumps: via rate kernel $Q_t$



## Results

### Experimental Summary

Experiments were conducted on:

- **Image generation** (CIFAR-10, ImageNet32)
- **Multimodal protein design** (combining structure and sequence modalities)

Key findings:

- **Jump models**, a novel class, exhibit reasonable generative capacity.
- **Markov superpositions** (e.g., combining flow and jump models) enhance FID scores.
- **Multimodal extensions** demonstrate state-of-the-art diversity in protein generation.

### Empirical Analysis

#### Image Generation

- **Jump models** perform adequately on CIFAR-10 and ImageNet32 but do not surpass finely tuned flow/diffusion models.
- Combining **jump + flow** via superpositions yields improved FID scores compared to individual models.
- Hybrid sampling (Euler for jumps, 2nd order ODE for flows) achieves the best results.

| Model                     | CIFAR-10 (FID) | ImageNet32 (FID) |
|---------------------------|----------------|------------------|
| Flow (Euler)              | 2.94           | 4.58             |
| Jump (Euler)              | 4.23           | 7.66             |
| Jump + Flow (Mixed)       | **2.36**       | **3.33**         |

#### Protein Generation

- Integration of SO(3) jump models into MultiFlow improves **diversity** and **novelty**.
- GM enables seamless **multimodal state space modeling**.

### Strengths

- **Theoretical Depth**: Grounded in stochastic process theory, offering solid derivations and generality.
- **Modality-Agnostic**: Applicable across $\mathbb{R}^d$, discrete spaces, and manifolds (e.g., SO(3)).
- **Unified Perspective**: Provides a principled framework encompassing various generative paradigms.
- **Innovative Potential**: Introduces new design principles, particularly in combining generative models across modalities and mechanisms (flow, diffusion, jumps).
- **Empirical Validation**: Demonstrates the practical viability of the framework through experiments on image and protein generation tasks.

### Limitations

- **Sampling Efficiency**: Sampling methods for jump models are less optimized compared to flows; only Euler methods are currently available.
- **Training Stability**: The impact of jump-induced discontinuities on training stability is not thoroughly explored.
- **Computational Overhead**: Learning and simulating general generators, especially those involving integration over jump measures, can be computationally intensive.
- **Comparative Analysis**: Limited comparison to autoregressive models, which are prevalent in many multimodal tasks (e.g., LLMs for text).



## Personal Take & Commentary

### Interpretation from My RL / Multi-Agent / LLM Agent Background

From the perspective of **Reinforcement Learning** and **LLM-based agents**, the concept of parameterizing **Markov generators** parallels the modeling of **transition dynamics** in model-based RL. Just as structured priors in environment modeling can enhance planning, Generator Matching offers a structured approach to define transition dynamics of distributions toward a data manifold.

In **multi-agent systems**, GM's framework for **composing models** (Markov superpositions, multimodal joints) suggests intriguing possibilities: combining agents with different generative priors or handling discrete-continuous hybrid state-action spaces.

Furthermore, the **Bregman divergence-based CGM loss** generalizes many losses used in score-based models and could be adaptable to RL-style divergence regularization objectives.

### Implications for Future Research

1. **RL Applications**: Exploring GM-inspired generators for learning flexible environment models in stochastic settings.
2. **LLM Agents**: Investigating the discrete diffusion perspective of language generation as an alternative to autoregressive decoding.
3. **Control as Generation**: Viewing control policies as generators transforming initial states to desired outcomes, potentially inspiring new training objectives for hierarchical policies.

### Critical Commentary

- The **mathematical rigor** is commendable, but practical challenges, especially in efficient sampling and scalable implementation, need further exploration. Real-world applications require robust inference schemes for jump models.
- The treatment of **multimodality** is theoretically sound but may oversimplify practical complexities, as real tasks involve intricate conditional dependencies.
- **Ablation studies** on the expressive benefits of modeling diffusion versus jump components separately would provide deeper insights.
- The assumption of access to the posterior $p_{1|t}(z|x)$ may pose challenges in complex domains, as approximations could become bottlenecks.


## 📌 TL;DR

**Generator Matching** presents a mathematically elegant and theoretically rich foundation for generative modeling using arbitrary Markov processes. It unifies existing methods under a cohesive framework and introduces powerful new design principles, particularly in combining and composing generative models across modalities and mechanisms. While practical challenges remain, especially concerning jump-based sampling and training, the framework opens exciting avenues for applications in RL, LLMs, and beyond.




# Conclusion

Both papers, though rooted in different problem domains, demonstrate how structured guidance—whether in the form of reward functions or stochastic generators—can profoundly influence generative performance. TACO operationalizes reinforcement learning with interpretable, biologically grounded reward shaping, showing how domain priors can lead to both better optimization and greater output diversity. Generator Matching, on the other hand, reimagines generative modeling itself as a problem of parameterizing transitions in a Markovian space, offering a versatile and modular approach to synthesizing data across modalities.

For researchers in reinforcement learning and LLM-based agents, these works offer complementary lessons. TACO echoes challenges in offline RL and exploration under uncertainty—issues also central to agent design in partially observable or data-scarce environments. Generator Matching suggests a formalism for model-based generation and planning, with intriguing implications for LLMs as stochastic transformers of information states, potentially inspiring new agent design strategies based on generator composition.

Finally, from a multi-agent systems perspective, both methods hint at future directions: in TACO, one could envision population-based or competitive optimization of DNA sequences; in Generator Matching, the compositionality of generators suggests a modular approach to modeling heterogeneous agent behaviors or hybrid continuous-discrete action spaces.

Together, these papers underscore a powerful trend: bridging generative modeling with policy design, whether for biology, vision, or intelligent agents. Their relevance spans from synthetic cells to synthetic cognition—united by a shared focus on controlled, high-fidelity generation in structured environments.