# Related Work Notes

This file collects detailed technical notes on prior work relevant to
Mixture-of-Experts (MoE) models, multimodal fusion, routing stability, and
robustness under heterogeneous conditions. Notes are organized strictly
per paper and preserve design-level insights, assumptions, and limitations.

---

## Scaling Vision with Sparse Mixture of Experts (V-MoE, NeurIPS 2021)

This work establishes sparse Mixture-of-Experts as a viable and effective
architecture for large-scale **vision models**, extending MoE beyond language.

### Core idea
- A Vision Transformer (ViT) backbone is modified by replacing selected dense
  feedforward (MLP) layers with sparse MoE layers.
- Routing is applied at the **patch-token level**: each image patch embedding
  is routed to a small subset of expert MLPs.
- MoE layers are not inserted everywhere; only certain layers are replaced.

### Architectural details
- Experts are standard MLP blocks.
- Routing uses sparse Top-k selection.
- MoE is introduced at layers with high representational capacity, not shallow
  layers.
- The authors argue that MoE is most effective when replacing blocks where
  additional capacity is actually useful.

### Key emphasis
- Primary focus is **scaling and efficiency**, not robustness or multimodal
  fusion.
- Demonstrates that sparse conditional computation can significantly increase
  model capacity without proportional compute cost.

### Relevance
- Establishes MoE as a legitimate design choice in vision.
- Supports the idea that **placement of MoE layers matters**.
- Less concerned with modality interaction, sensor reliability, or
  environmental variation.

---

## ST-MoE: Designing Stable and Transferable Sparse Expert Models (2022)

This paper focuses on the **stability and robustness** of sparse MoE routing,
with particular attention to routing collapse, numerical sensitivity, and
behavior under distribution shift.

### Routing instability and multiplicative sensitivity
- Every additional **multiplicative interaction** inside the routing mechanism
  increases sensitivity and destabilizes expert selection.
- Sources of instability include:
  - Deep or complex gating MLPs
  - Strong nonlinear interactions between context signals
  - Multiplicative fusion of modality features and context inside gates

### Important distinction
- Multiplicative interactions **inside expert networks** can improve expressivity
  without destabilizing routing, because routing decisions remain unchanged.
- Multiplicative interactions **inside routers** amplify small perturbations and
  lead to brittle expert assignment.

### Design takeaways
- Gating networks should be kept **simple and shallow**.
- Context signals should **bias** routing rather than dominate it.
- Additive conditioning of router logits is more stable than multiplicative
  modulation.
- Routing instability across conditions (e.g., lighting or weather) may reflect
  sensitivity rather than meaningful adaptation.

### Router z-loss
- Introduces a “router z-loss” that penalizes large router logits.
- Keeps logits in a soft regime and prevents early hard expert assignment.
- Reduces brittleness caused by sharp softmax outputs and Top-k thresholds.

### Numerical precision issues
- Sparse MoEs are more sensitive to numerical roundoff than dense models because:
  1. Softmax outputs are used for **discrete Top-k selection**
  2. Softmax outputs are also used to **scale expert outputs multiplicatively**
- This creates two instability channels:
  - Selection instability (expert chosen or not)
  - Scaling instability (expert contribution magnitude)

- Smaller logits mitigate this issue, as large values have coarser representation
  in finite precision.

### Training regime
- Sparse MoEs benefit from noisier optimization:
  - smaller batch sizes
  - higher learning rates
  - stronger regularization

---

## Efficient Early-Fusion Pre-training with Modality-Aware Experts (MoMa, 2024)

MoMa studies MoE architectures in **early-fusion multimodal models**, focusing
on efficiency and expert utilization.

### Setup: early fusion
- Image and text inputs are represented as discrete tokens.
- Tokens are processed jointly by a shared Transformer backbone.
- Full self-attention operates over the combined token sequence.
- Tokens retain modality identity (image vs text).

### Problem identified
- Applying sparse MoE layers naively to early-fused token sequences is inefficient.
- Routing heterogeneous tokens (image + text) into a single expert pool leads to:
  - poor expert utilization
  - redundant computation
  - degraded scaling efficiency

### MoMa design
- Experts are divided into **modality-specific groups**.
- Image tokens are routed only to image experts.
- Text tokens are routed only to text experts.
- Routing is learned **within each modality group**.

### Benefits claimed
- Improved efficiency by avoiding irrelevant expert computation.
- Enhanced specialization within each modality.
- Cross-modal interaction is preserved through shared self-attention layers
  outside the MoE blocks.

### Limitations
- Fusion order is fixed by design (fusion occurs before MoE).
- Alternative fusion–MoE placements (late fusion, fusion-then-MoE) are not explored.
- Focus is on scaling and efficiency, not robustness under modality degradation.

---

## On the Representation Collapse of Sparse Mixture of Experts (Chi et al.)

This work analyzes a failure mode specific to **stacked sparse MoE layers**.

### Central observation
- Sparse routing induces a self-reinforcing feedback loop:
  1. Token is routed to expert e
  2. Expert e adapts parameters to fit the token
  3. Token representation shifts toward expert e
  4. Router becomes more confident assigning token to e
  5. Loop repeats, leading to representation collapse

### Proposed solution
- Project token representations to a lower-dimensional space.
- Apply L2 normalization to both tokens and expert weights.
- Compute routing similarity on a hypersphere.

### Scope relevance
- Collapse effects are most pronounced with **multiple stacked MoE layers**.
- With a single MoE layer, representation collapse is less severe and likely
  negligible.

---

## Multimodal Sensor Fusion for Autonomous Driving: A Survey (Sensors, 2025)

This survey provides a structured overview of multimodal fusion strategies and
their challenges in real-world autonomous driving.

### Modalities and limitations
- Cameras: dense semantic and texture information; degrade in low light and fog.
- LiDAR: accurate 3D geometry; sensitive to weather and expensive.
- Radar: robust range and velocity; sparse spatial resolution.

### Fusion categories
- Early fusion (sensor-level):
  - combines raw data
  - suffers from alignment and synchronization challenges
- Mid-level fusion (feature-level):
  - combines intermediate features
  - widely used in modern systems
- Late fusion (decision-level):
  - combines outputs of modality-specific branches
  - simpler but weaker synergy

### Robustness challenges
- Sensor degradation due to environment or aging
- Calibration errors and misalignment
- Domain shifts across geography and weather

### Conceptual point
- Classical fusion often assumes conditional independence given the scene.
- Real-world perception exhibits conditional correlation due to shared structure
  and unmodeled factors.
- Adaptive fusion mechanisms are therefore required.

---

## Multi-Modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion (ICCV 2023)

This paper studies adaptive image fusion under varying modality reliability.

### Core observation
- Modality informativeness varies with conditions:
  - visible texture dominates under good lighting
  - infrared dominates under low-light conditions
- Fixed fusion strategies suppress useful signals when reliability shifts.

### MoE formulation
- Uses a gated MoE to condition fusion behavior on the input.
- MoE is used for **adaptive fusion**, not capacity scaling.

### Failure of unconstrained routing
- When experts all see similar inputs and routing is learned purely from features:
  - specialization is opaque
  - experts mix modalities and scales unpredictably
  - interpretability and effectiveness suffer

### Structured expert design
- Experts are explicitly differentiated:
  - local experts focus on fine-grained spatial detail
  - global experts capture global context
- Some experts are tied to specific modality streams.
- Routing is conditioned on auxiliary context signals (attention maps).

### Key result
- Imposing architectural constraints on expert roles leads to better performance.
- Removing local/global experts or replacing them with attention-only mechanisms
  consistently degrades performance.

---

## Mixpert: Mitigating Multimodal Learning Conflicts with Efficient Mixture-of-Vision-Experts (CVPR 2024)

Mixpert analyzes multimodal learning from an optimization perspective.

### Optimization conflict
- Joint training across heterogeneous visual domains induces gradient conflict.
- Shared parameters receive competing updates, leading to compromised solutions.

### Late specialization strategy
- Early layers remain shared to extract general features.
- Deeper layers and the projection head are replicated into experts.
- Only a small fraction of the network is expertized.

### Training procedure
- A shared model is trained first.
- Late layers are cloned to form experts.
- Experts are fine-tuned independently on domain-specific data.
- Shared backbone and other experts are frozen.

### Routing and inference
- No routing during expert specialization.
- Routing is applied only at inference time.
- When routing confidence is low, inference falls back to a shared, versatile expert.

### Implications
- Meaningful specialization does not emerge automatically from end-to-end MoE.
- Structure and staged training can yield more stable and interpretable experts.
- Conditional computation can be efficient if confined to late layers.

---

## Mixpert: additional inference-time insight

- Routing predicts domain from shared features.
- When confidence gap between experts is small, a generalist expert is used.
- This acts as a robustness safeguard against uncertain routing decisions.

---

## MoVA: Adapting Mixture of Vision Experts to Multimodal Context (NeurIPS 2024)

MoVA addresses expert selection when multiple pretrained vision encoders are available.

### Problem
- Any single vision encoder is biased toward certain domains.
- Naive fusion of multiple encoders degrades performance due to irrelevant or
  misleading representations.

### Expert definition
- Experts are pretrained vision encoders with known inductive biases.
- Expert roles are fixed and predefined.

### Routing philosophy
- Routing is coarse-grained and semantic.
- Uses low-resolution visual tokens, task instructions, and textual descriptions
  of expert capabilities.
- Routing is **not** optimized end-to-end via downstream gradients.

### Separation of concerns
- Expert selection is treated as a discrete decision problem.
- Expert fusion is handled separately via a soft MoE adapter.
- Avoids gradient-driven feedback loops between routing and representation learning.

### Regularization stance
- No per-batch load balancing.
- Expert usage is allowed to be sparse and highly imbalanced.
- Only weak global regularization is used to prevent permanent collapse.

### Key insight
- Imbalance often reflects correct contextual relevance, not failure.
- For many inputs, only one or two experts should dominate.

---

## Mixpert, MoVA, and adverse conditions (note)

- Under adverse conditions, one modality or encoder can become actively misleading.
- Forcing uniform usage reintroduces corrupted signals.
- Conditional suppression (not just downweighting) is required for robustness.
