Activation Science of Large Language Models

  A Systematic Framework for Understanding Representation Dynamics

  ---
  1. FOUNDATIONAL THEORY

  1.1 The Core Object: Activation Trajectories

  A large language model transforms discrete token sequences into continuous vector trajectories through a stack of residual layers. The
  fundamental object of study is the activation trajectory — the path a representation takes through high-dimensional space as it traverses
  layers (depth) and token positions (time).

  We formalize this with two dynamical equations that govern all activation behavior:

  Depth dynamics (residual computation across layers):

  $$h_{\ell+1}^{(t)} = h_\ell^{(t)} + f_\ell(h_\ell^{(t)}, {h_\ell^{(s)}}_{s \leq t})$$

  where $f_\ell$ is the combined attention + MLP update at layer $\ell$, and the dependence on ${h_\ell^{(s)}}_{s \leq t}$ captures
  cross-token information flow via attention.

  Temporal dynamics (autoregressive generation across tokens):

  $$h_\ell^{(t+1)} = G_\ell(\text{embed}(x_{t+1}), {h_\ell^{(s)}}_{s \leq t})$$

  where $G_\ell$ encapsulates the full forward pass up to layer $\ell$, and $x_{t+1}$ is selected from the output distribution at the
  previous step.

  These two equations define a coupled depth-time dynamical system. The depth axis performs computation; the time axis accumulates context.
  Activation Science studies the geometry, dynamics, and computational content of trajectories in this system.

  1.2 Five Theoretical Lenses

  We organize the theory through five complementary perspectives, each illuminating different aspects of the same underlying system.

  Lens 1: Representation Geometry

  Hidden states occupy a tiny fraction of their ambient $\mathbb{R}^d$ space. The intrinsic geometry of this occupied manifold — its
  dimensionality, curvature, anisotropy, and cluster structure — reveals how the model organizes information. Key questions: What is the
  effective dimensionality at each layer? Do representations concentrate on low-dimensional manifolds? How does geometry differ across
  linguistic tasks?

  Core prediction: Representations undergo a geometric transformation across layers — from high-dimensional, approximately isotropic
  distributions in early layers to low-dimensional, anisotropic distributions in late layers, reflecting compression toward the output
  vocabulary.

  Lens 2: Residual Computation

  The residual stream $h_\ell$ is a communication channel. Each layer reads from it, computes an update $f_\ell(h_\ell)$, and writes the
  result back. The magnitude, direction, and information content of these updates characterize what each layer contributes. Key questions:
  Which layers make large vs. small changes? Do early layers perform different types of computation than late layers? When do layers become
  redundant?

  Core prediction: Layer contributions follow a characteristic profile — large updates in early layers (syntactic processing), moderate
  updates in middle layers (semantic composition), and small, precise updates in late layers (output refinement) — with task-dependent
  deviations from this baseline.

  Lens 3: Information Flow and Decision Formation

  At every layer, the hidden state implicitly encodes a distribution over next tokens. The logit lens and its refinements track how this
  distribution evolves. Key questions: When does the model "decide" the next token? Is decision formation gradual or punctuated? Do
  different token types (content words, function words, punctuation) crystallize at different layers?

  Core prediction: Decision formation exhibits a crystallization pattern — the output distribution transitions from diffuse (high entropy)
  to concentrated (low entropy) at a layer that depends on the predictability and syntactic role of the token. Easy predictions crystallize
  early; difficult predictions remain uncertain until late layers.

  Lens 4: Autoregressive State Dynamics

  During generation, the model's hidden states at each token position form a trajectory through representation space. This trajectory has
  dynamical properties — it may converge to attractors, exhibit oscillations, or undergo regime transitions. Key questions: Are there stable
   generation regimes? Do different text types produce qualitatively different trajectories? Can we detect mode shifts in generation?

  Core prediction: Generation trajectories cluster into a small number of dynamical regimes (e.g., fluent continuation, uncertain
  exploration, template completion) that are detectable from hidden-state geometry and that correlate with observable text properties.

  Lens 5: Cross-Sequence Universality

  Different input sequences processed by the same model may share geometric structure at certain layers — a form of representational
  universality. Key questions: Which layers produce representations that are most invariant to surface form? Do semantically similar inputs
  converge to shared subspaces? Is there a canonical "meaning space" at intermediate layers?

  Core prediction: Middle layers exhibit maximum cross-sequence alignment for semantically similar inputs, while early layers are
  surface-form-dependent and late layers are output-distribution-dependent. This creates a semantic bottleneck at intermediate depth.

  1.3 Structural Expectations

  From these lenses, we derive concrete structural expectations about activation space:

  1. Layer-wise phase structure: The layer stack partitions into functional phases (encoding → composition → prediction) with detectable
  boundaries.
  2. Low-rank residual updates: Individual layer contributions occupy low-dimensional subspaces relative to the ambient dimension.
  3. Token-type stratification: Different syntactic/semantic categories occupy distinguishable regions of activation space.
  4. Temporal coherence: Successive decode steps produce hidden states with high cosine similarity (confirmed by Experiment 0), but with
  structure in the deviations.
  5. Dataset-dependent signatures: Task domain shapes activation geometry in predictable ways — mathematical reasoning vs. narrative
  generation vs. factual recall produce different geometric signatures.

  ---
  2. TAXONOMY OF ACTIVATION PHENOMENA

  We organize activation behavior into seven categories. Each category identifies a distinct class of phenomena, the theoretical lens that
  motivates it, and the experimental signatures that would confirm or refute its predictions.

  2.1 Geometry — The Shape of Representation Space

  What it represents: The static geometric structure of hidden-state distributions at fixed layers and token positions.

  Why it matters: Geometry constrains computation. A representation that lies on a low-dimensional manifold can only encode information
  about a limited number of features. Anisotropy (directional bias) indicates that certain directions in activation space are privileged.
  The effective dimensionality at each layer reveals how many independent features the model maintains.

  Key phenomena:
  - Anisotropy gradient: Representations become increasingly anisotropic (directionally biased) in deeper layers.
  - Dimensional collapse: Effective rank decreases across layers as the model compresses toward output.
  - Spectral structure: The singular value spectrum of activation matrices reveals whether representations are smooth (gradual spectral
  decay) or clustered (spectral gaps).
  - Outlier dimensions: A small number of activation dimensions carry disproportionate variance, distorting similarity metrics.

  2.2 Dynamics — How Representations Evolve

  What it represents: The temporal evolution of hidden states during autoregressive generation.

  Why it matters: LLMs are generative systems. The trajectory of hidden states across decode steps reveals the model's internal "state of
  mind" during generation — whether it is confidently continuing, exploring alternatives, or transitioning between modes.

  Key phenomena:
  - Temporal coherence: Successive decode steps produce highly similar hidden states (baseline from Experiment 0).
  - Drift rate variation: Some layers show faster representational change than others during generation.
  - Regime transitions: Abrupt changes in hidden-state trajectory indicating shifts in generation mode.
  - Attractor dynamics: Generation may converge toward fixed points or limit cycles in representation space.

  2.3 Computation — What Each Layer Contributes

  What it represents: The functional role of individual layers, characterized by the magnitude, direction, and information content of their
  residual updates.

  Why it matters: Understanding layer-wise computation is the bridge between activation analysis and mechanistic interpretability. If we can
   characterize what information each layer adds, modifies, or preserves, we can build a functional map of the model.

  Key phenomena:
  - Update magnitude profile: The norm of $f_\ell(h_\ell)$ across layers reveals which layers make large vs. small contributions.
  - Update orthogonality: Whether successive layer updates are orthogonal (adding independent information) or aligned (refining a shared
  direction).
  - Residual saturation: Late layers may make negligible updates, suggesting the computation is effectively complete before the final layer.
  - Phase boundaries: Sharp transitions in update properties that demarcate functional phases of the network.

  2.4 Decision Formation — The Path to Output

  What it represents: How the model's implicit prediction of the next token evolves across layers.

  Why it matters: This is the most direct link between internal representations and observable behavior. Tracking when and how the output
  distribution crystallizes reveals the computational depth required for different predictions.

  Key phenomena:
  - Entropy reduction curve: The entropy of the logit-lens distribution decreases across layers, with the rate depending on prediction
  difficulty.
  - Early crystallization: For predictable tokens, the correct prediction appears in early layers.
  - Late disambiguation: For ambiguous contexts, multiple candidates persist until late layers.
  - Prediction stability: Once the top prediction stabilizes, it rarely changes — a form of computational commitment.

  2.5 Representation Reuse — Activation Memory

  What it represents: The degree to which the model reuses representations from earlier token positions when computing the current token's
  hidden state.

  Why it matters: This reveals the model's internal memory structure — which prior tokens are "consulted" at each layer, and whether this
  consultation pattern reflects linguistic structure (e.g., coreference, syntactic dependencies).

  Key phenomena:
  - Similarity decay curves: Cosine similarity between current and prior tokens as a function of distance (confirmed structured decay in
  Experiment 0).
  - Positional anchoring: Certain positions (e.g., first token, recent tokens) serve as persistent reference points.
  - Layer-dependent memory: Early layers may reference syntactically local tokens; late layers may reference semantically relevant distant
  tokens.
  - Dataset-dependent memory patterns: Factual recall tasks show different reference patterns than narrative continuation.

  2.6 Information Routing — Cross-Token Communication

  What it represents: How information flows between token positions via attention, and how this flow shapes the representational landscape.

  Why it matters: Attention is the primary mechanism for cross-token communication. The patterns of information routing determine which
  tokens influence which, creating the effective computational graph of the model.

  Key phenomena:
  - Attention sink formation: Certain token positions (typically the first token) accumulate attention weight disproportionately.
  - Routing specialization: Different attention heads route information for different purposes (syntactic vs. semantic vs. positional).
  - Information bottlenecks: Layers where cross-token communication is concentrated, creating computational chokepoints.
  - Causal influence structure: The actual (not just attentional) influence of each token on downstream representations.

  2.7 State Regimes — Macroscopic Generation Modes

  What it represents: Qualitatively distinct modes of generation that are detectable from the macroscopic properties of hidden-state
  trajectories.

  Why it matters: If generation regimes exist, they provide a high-level vocabulary for describing LLM behavior — analogous to phases of
  matter in physics. Detecting regime transitions during generation could enable real-time monitoring of model behavior.

  Key phenomena:
  - Regime clustering: Hidden-state trajectories cluster into distinct modes in PCA or UMAP projections.
  - Regime transitions: Detectable shifts between clusters during generation.
  - Regime-behavior correlation: Different regimes correlate with observable text properties (fluency, uncertainty, creativity).
  - Cross-dataset regime universality: The same regimes appear across different tasks, suggesting model-intrinsic (rather than
  task-specific) generation modes.

  ---
  3. EXPERIMENTAL FRAMEWORK

  Each experiment is mapped to one or more taxonomy categories, with explicit hypotheses, measurements, metrics, and expected signatures. We
   label the six experiments in our pipeline as E0–E5, and propose additional experiments E6–E8.

  E0: Decode-Time Cosine Similarity [COMPLETED]

  Taxonomy: Dynamics, Representation Reuse

  Hypothesis: During autoregressive generation, hidden states at each layer exhibit structured similarity to prior token positions, with the
   similarity pattern depending on layer depth, token distance, and task domain.

  Measurements:
  - Cosine similarity between each decode-step hidden state and all prior hidden states, per layer
  - Top-k most similar reference positions per layer per decode step

  Metrics:
  - Mean, median, max, std of per-layer similarity
  - Fraction of pairs above thresholds (0.70, 0.90, 0.95, 0.98)
  - Reference position distribution (distance histogram of top-k references)
  - Similarity decay as function of token distance

  Data Collection:
  - 7 datasets × 4 context lengths × 2 decode lengths × 3 prompts = 168 experiments
  - Model: Qwen2.5-32B-Instruct
  - Storage: dual Parquet schema (top-k references + aggregate statistics)

  Expected Signatures (confirmed by results):
  - High baseline similarity (>0.90) in most layers, indicating strong temporal coherence
  - Distance-dependent decay with dataset-specific profiles
  - Layer-dependent reference patterns (early layers: local references; late layers: broader references)
  - Attention-sink effects at position 0

  Status: Complete. 168/168 experiments, 27 analysis plots generated.

  ---
  E1: Residual Stream Decomposition

  Taxonomy: Computation

  Hypothesis: Layer contributions follow a characteristic magnitude and directional profile that partitions the network into functional
  phases, with phase boundaries that shift depending on task domain.

  Measurements:
  - Residual update $\Delta_\ell = h_{\ell+1} - h_\ell$ for each layer during both prefill and decode
  - Cosine similarity between successive updates: $\cos(\Delta_\ell, \Delta_{\ell+1})$
  - Projection of updates onto the residual stream direction

  Metrics:
  - Update norm $|\Delta_\ell|$ per layer (absolute and relative to $|h_\ell|$)
  - Update direction cosine: $\cos(\Delta_\ell, h_\ell)$ — is the update aligned with or orthogonal to the current state?
  - Cumulative explained variance: what fraction of the final representation is attributable to updates from layers $1, \ldots, \ell$?
  - Phase transition detection: derivative of update norm profile

  Data Collection:
  - Same 7 datasets, context lengths, prompts as E0
  - Both prefill and decode phases
  - Storage: per-layer update statistics in Parquet

  Expected Signatures:
  - Three-phase update profile: large early updates, moderate middle updates, small late updates
  - Prefill vs. decode divergence: prefill updates are larger (processing new information); decode updates are smaller (incremental
  prediction)
  - Orthogonality increase: later layer updates are more orthogonal to the residual stream, adding new information rather than amplifying
  existing directions
  - Task-dependent phase boundaries: reasoning tasks (GSM8K) may show extended middle-phase computation

  ---
  E2: Logit Lens (Intermediate Vocabulary Projection)

  Taxonomy: Decision Formation

  Hypothesis: The model's implicit next-token prediction crystallizes at a layer that depends on token predictability and syntactic role,
  with a characteristic entropy reduction curve.

  Measurements:
  - Apply the unembedding matrix $W_U$ to hidden states at each layer: $\text{logits}\ell = W_U \cdot h\ell^{(t)}$
  - Convert to probability distribution via softmax
  - Track the rank and probability of the final predicted token at each layer

  Metrics:
  - Entropy of the logit-lens distribution at each layer
  - KL divergence between layer-$\ell$ distribution and final-layer distribution
  - Crystallization layer: earliest layer where the top-1 prediction matches the final prediction
  - Top-k prediction stability: how the top-5 candidate set changes across layers

  Data Collection:
  - Same 7 datasets, context lengths, prompts
  - Decode phase, per-token per-layer measurements
  - Storage: per-layer entropy, top-k predictions, rank of final token

  Expected Signatures:
  - Monotonic entropy decrease across layers (with possible non-monotonicities at phase boundaries)
  - Function words crystallize early (layers 5–15); content words crystallize late (layers 25–50)
  - Mathematical reasoning tokens (GSM8K) show delayed crystallization relative to narrative tokens
  - KL divergence follows an exponential decay with a task-dependent rate constant

  ---
  E3: Representation Geometry (Anisotropy, Effective Rank, SVD Spectrum)

  Taxonomy: Geometry

  Hypothesis: The geometric structure of activation distributions varies systematically across layers, with a transition from approximately
  isotropic, high-dimensional distributions in early layers to anisotropic, low-dimensional distributions in late layers.

  Measurements:
  - Collect hidden-state matrices $H_\ell \in \mathbb{R}^{T \times d}$ (tokens × hidden dimension) at each layer
  - Compute SVD: $H_\ell = U \Sigma V^T$

  Metrics:
  - Effective rank: $\text{erank}(H_\ell) = \exp(-\sum_i p_i \log p_i)$ where $p_i = \sigma_i / \sum_j \sigma_j$
  - Participation ratio: $(\sum_i \sigma_i^2)^2 / \sum_i \sigma_i^4$
  - Anisotropy: average cosine similarity between random pairs of hidden states
  - Spectral decay rate: fit $\sigma_i \propto i^{-\alpha}$ and extract $\alpha$
  - Top-k variance explained: fraction of variance captured by top $k$ singular vectors

  Data Collection:
  - Hidden states from all token positions at each layer
  - Both prefill completion and decode phases
  - Multiple context lengths to test length dependence

  Expected Signatures:
  - Effective rank decreases from early to late layers (dimensional collapse)
  - Anisotropy increases with depth (directional concentration)
  - Spectral decay rate $\alpha$ increases with depth (steeper spectrum = lower effective dimensionality)
  - Dataset-dependent geometry: mathematical reasoning may maintain higher dimensionality in middle layers
  - Outlier dimensions detectable as spectral gaps in the singular value distribution

  ---
  E4: Cross-Sequence Representation Alignment

  Taxonomy: Cross-Sequence Universality

  Hypothesis: Semantically similar sequences converge to shared representational subspaces at intermediate layers, creating a semantic
  bottleneck, while early layers are surface-form dependent and late layers are output-distribution dependent.

  Measurements:
  - For pairs of sequences, extract hidden-state matrices $H_\ell^{(A)}$ and $H_\ell^{(B)}$ at each layer
  - Compute alignment metrics between these matrices

  Metrics:
  - Linear CKA: centered kernel alignment measuring linear representational similarity
  - Procrustes distance: minimum-rotation distance between representations (after alignment)
  - Subspace overlap: $\frac{1}{k}\sum_{i=1}^k |V_B^T v_i^A|^2$ — overlap between top-$k$ principal subspaces
  - Centroid cosine: cosine similarity between mean representations
  - Shared-token cosine: for tokens appearing in both sequences, direct hidden-state comparison

  Data Collection:
  - Within-dataset pairs (same domain, different prompts)
  - Cross-dataset pairs (different domains)
  - Same prompt with different context lengths
  - Storage: per-layer per-metric per-pair results

  Expected Signatures:
  - CKA peaks at intermediate layers (semantic bottleneck)
  - Same-dataset pairs show higher alignment than cross-dataset pairs at all layers
  - Procrustes distance is minimized at the layer where semantic representations are most universal
  - Subspace overlap reveals shared principal directions that encode domain-invariant features
  - Late-layer alignment reflects output distribution similarity rather than semantic similarity

  ---
  E5: Activation State Detection (Generation Trajectories)

  Taxonomy: State Regimes, Dynamics

  Hypothesis: Autoregressive generation produces hidden-state trajectories that cluster into a small number of dynamical regimes, detectable
   via PCA projections, with regime transitions correlating to observable properties of the generated text.

  Measurements:
  - Track hidden states at selected layers across all decode steps
  - Project onto principal components (fitted on the full trajectory)
  - Compute trajectory statistics: velocity, curvature, cluster membership

  Metrics:
  - PCA trajectory: positions in top-20 principal component space at each decode step
  - Trajectory velocity: $|h_\ell^{(t+1)} - h_\ell^{(t)}|$ per step
  - Trajectory curvature: angular change in velocity vector
  - Cluster membership: k-means or HDBSCAN clustering of trajectory points
  - Regime persistence: average duration of stay in each cluster
  - Transition matrix: empirical probabilities of regime-to-regime transitions

  Data Collection:
  - Extended decode (256 tokens) for trajectory analysis
  - 9 evenly spaced layers for computational efficiency
  - Same 7 datasets, multiple prompts
  - Storage: per-step PCA coordinates, velocities, cluster labels

  Expected Signatures:
  - 3–7 distinct trajectory clusters (regimes) per layer
  - Regime transitions correlate with syntactic boundaries (sentence endings, clause boundaries)
  - Different datasets produce different regime distributions (e.g., GSM8K shows a "computation" regime absent in narrative tasks)
  - Cross-layer regime coherence: transitions at one layer predict transitions at others
  - Attractor-like behavior: trajectories converge toward cluster centers during fluent generation

  ---
  E6: Causal Layer Intervention [PROPOSED NEW]

  Taxonomy: Computation, Information Routing

  Hypothesis: Individual layers make causally necessary contributions to output quality that can be quantified by measuring the effect of
  ablation (zeroing) or clamping (fixing) the residual update at each layer.

  Measurements:
  - Run normal forward pass, record all hidden states
  - For each layer $\ell$, re-run with $\Delta_\ell = 0$ (skip the layer) and measure output change
  - For each layer $\ell$, re-run with $\Delta_\ell$ clamped to its mean value across tokens

  Metrics:
  - Output KL divergence: $D_{KL}(p_\text{original} | p_\text{ablated})$ per layer
  - Prediction flip rate: fraction of tokens where top-1 prediction changes
  - Downstream propagation: how ablation at layer $\ell$ affects hidden states at layers $\ell+1, \ldots, L$
  - Causal importance score: integrated effect of ablation across all output positions

  Expected Signatures:
  - A small number of "critical" layers whose ablation causes large output changes
  - Critical layers coincide with phase boundaries identified in E1
  - Early-layer ablation propagates widely; late-layer ablation has localized effects
  - Task-dependent criticality: different layers are critical for different task types

  ---
  E7: Representation Perturbation Sensitivity [PROPOSED NEW]

  Taxonomy: Geometry, Dynamics

  Hypothesis: The model's sensitivity to activation perturbations varies across layers and directions, revealing the functional geometry of
  representation space — which directions matter for computation and which are noise-tolerant.

  Measurements:
  - At each layer, perturb the hidden state: $h_\ell' = h_\ell + \epsilon \cdot v$ for various perturbation directions $v$
  - Measure output change as a function of perturbation magnitude and direction

  Metrics:
  - Directional sensitivity: output change per unit perturbation along each singular vector direction
  - Sensitivity spectrum: sorted sensitivity values across directions
  - Functional dimensionality: number of directions with sensitivity above a threshold
  - Sensitivity-geometry correlation: relationship between singular value magnitude and directional sensitivity

  Expected Signatures:
  - Sensitivity is concentrated in a low-dimensional subspace (functional directions)
  - High-singular-value directions are not always high-sensitivity directions (geometric importance ≠ functional importance)
  - Late layers are more sensitive (small perturbations cause large output changes)
  - Sensitivity landscape is task-dependent: reasoning tasks have higher sensitivity in middle layers

  ---
  E8: Token-Type Stratified Analysis [PROPOSED NEW]

  Taxonomy: Geometry, Decision Formation, Representation Reuse

  Hypothesis: Different syntactic and semantic token categories (function words, content words, punctuation, numbers, named entities) occupy
   distinct regions of activation space, with the degree of separation varying across layers.

  Measurements:
  - POS-tag and NER-classify all tokens in the input
  - Group hidden states by token category at each layer
  - Compute within-group and between-group statistics

  Metrics:
  - Category centroid distances: pairwise cosine distance between category centroids per layer
  - Category separability: Fisher discriminant ratio (between-class variance / within-class variance)
  - Layer of maximum separation: the layer at which category centroids are most spread
  - Category-specific geometry: effective rank and anisotropy computed per token category

  Expected Signatures:
  - Token categories are poorly separated in early layers, maximally separated in middle layers, and partially re-merged in late layers (as
  representations compress toward output logits)
  - Function words form a tighter cluster than content words (lower within-class variance)
  - Named entities and numbers have distinctive geometric signatures in middle layers
  - The layer of maximum separation coincides with the semantic bottleneck identified in E4

  ---
  4. MEASUREMENT AND ANALYSIS METHODS

  4.1 Spectral Analysis

  Tools: Singular Value Decomposition (SVD), eigendecomposition of covariance matrices.

  What it reveals: The intrinsic dimensionality and directional structure of activation distributions. The singular value spectrum is a
  fingerprint of representational geometry — a flat spectrum indicates distributed representations; a steep spectrum indicates concentration
   on a few principal directions.

  Key derived quantities:
  - Effective rank (Shannon entropy of normalized singular values)
  - Participation ratio (inverse Herfindahl index of squared singular values)
  - Spectral gap locations (potential cluster boundaries)
  - Variance explained curves (information compression)

  Application across experiments: E3 (primary), E4 (subspace overlap), E5 (PCA trajectory), E7 (sensitivity spectrum).

  4.2 Subspace Methods

  Tools: Principal Component Analysis, Procrustes alignment, Grassmannian distance, subspace intersection.

  What it reveals: Whether different conditions (layers, datasets, token types) share representational structure. Subspace overlap
  quantifies the degree to which two representations use the "same directions" in activation space.

  Key derived quantities:
  - Principal angles between subspaces
  - Procrustes rotation matrices (alignment transformations)
  - Shared variance in common subspace
  - Grassmannian distance (metric on subspace manifold)

  Application across experiments: E4 (primary), E3 (cross-layer subspace continuity), E8 (category subspaces).

  4.3 Information-Theoretic Measures

  Tools: Shannon entropy, KL divergence, mutual information estimation.

  What it reveals: The information content of representations and how it transforms across layers. Entropy of the logit-lens distribution
  measures prediction uncertainty; KL divergence between layers measures information gain per layer; mutual information between layers
  measures redundancy.

  Key derived quantities:
  - Logit-lens entropy profile (information compression across layers)
  - Layer-to-layer KL divergence (information gain per step)
  - Cross-layer mutual information (redundancy structure)
  - Representation entropy (approximated via nearest-neighbor or binning methods)

  Application across experiments: E2 (primary), E0 (similarity entropy), E5 (trajectory entropy).

  4.4 Dynamical Systems Analysis

  Tools: Phase space reconstruction, Lyapunov exponents, recurrence analysis, regime detection.

  What it reveals: The temporal structure of generation trajectories. Attractors indicate stable generation modes; high Lyapunov exponents
  indicate sensitivity to initial conditions (chaotic behavior); recurrence indicates periodic or quasi-periodic dynamics.

  Key derived quantities:
  - Trajectory velocity and acceleration profiles
  - Approximate Lyapunov exponents (divergence rates of nearby trajectories)
  - Recurrence quantification analysis (RQA) metrics
  - Regime transition detection via change-point analysis

  Application across experiments: E5 (primary), E0 (temporal dynamics), E1 (update dynamics).

  4.5 Causal Intervention Methods

  Tools: Activation patching, mean ablation, directional ablation, interchange intervention.

  What it reveals: The causal rather than merely correlational role of specific activations. Patching identifies which components are
  necessary for a particular output; interchange intervention identifies which components carry specific information.

  Key derived quantities:
  - Causal effect magnitude (output change per intervention)
  - Causal graph (which components influence which outputs)
  - Information localization (where specific facts or features are stored)
  - Necessity vs. sufficiency scores for each component

  Application across experiments: E6 (primary), E7 (perturbation sensitivity).

  4.6 Representation Similarity Metrics

  Tools: CKA (linear and kernel), Representational Similarity Analysis (RSA), Canonical Correlation Analysis (CCA), centered correlation.

  What it reveals: Whether two sets of representations (from different layers, models, or inputs) encode the same information, potentially
  in different formats. CKA is invariant to orthogonal transformation and isotropic scaling, making it robust to superficial
  representational differences.

  Key derived quantities:
  - CKA similarity matrix (layers × layers, or conditions × conditions)
  - CCA canonical correlations (shared information dimensions)
  - RSA second-order similarity (do two representations preserve the same distance structure?)

  Application across experiments: E4 (primary), E3 (cross-layer CKA).

  ---
  5. POTENTIAL DISCOVERIES

  5.1 Representation Collapse and Recovery

  Prediction: Representations undergo dimensional collapse in late layers as they compress toward the output vocabulary, but specific task
  types (reasoning, multi-step computation) may show collapse delay or mid-network expansion where the model temporarily increases effective
   dimensionality to perform complex operations.

  Significance: Would demonstrate that the model dynamically allocates representational capacity based on computational demands, analogous
  to working memory expansion during difficult cognitive tasks.

  Evidence path: E3 (effective rank across layers) + E2 (entropy profiles) + cross-task comparison.

  5.2 Universal Semantic Bottleneck

  Prediction: There exists a range of intermediate layers where representations from semantically similar inputs (regardless of surface
  form) converge to a shared subspace. This bottleneck is the model's internal "meaning space" — maximally invariant to syntax and maximally
   informative about semantics.

  Significance: Would identify the layer range where the model has most completely transformed surface-level token sequences into abstract
  semantic representations, with implications for transfer learning and representation extraction.

  Evidence path: E4 (cross-sequence CKA peak) + E8 (category separation peak) + E3 (geometric transition).

  5.3 Generation Regime Taxonomy

  Prediction: Autoregressive generation is not a uniform process. Hidden-state trajectories reveal 3–7 distinct generation regimes —
  including at minimum: (a) fluent continuation (low velocity, high coherence), (b) uncertain exploration (high velocity, low coherence),
  (c) structural transition (abrupt direction change at sentence/paragraph boundaries), and (d) template completion (very low velocity,
  near-constant hidden state). For reasoning models, an additional (e) computational deliberation regime may be detectable.

  Significance: Would provide a principled vocabulary for describing LLM behavior during generation, with potential applications to
  generation quality monitoring, uncertainty estimation, and controllable generation.

  Evidence path: E5 (trajectory clustering) + E0 (temporal coherence patterns) + E2 (entropy dynamics).

  5.4 Layer Phase Transitions

  Prediction: The layer stack is not a smooth gradient of computation. Instead, there are sharp transitions — layers where update magnitude,
   direction, spectral properties, or logit-lens predictions change abruptly. These transitions demarcate functional phases of the network
  (e.g., syntactic parsing → semantic composition → output prediction).

  Significance: Would establish that deep networks organize into functionally distinct modules despite uniform architecture, with
  implications for model pruning, layer grafting, and efficient inference.

  Evidence path: E1 (update magnitude/direction profiles) + E3 (spectral transitions) + E6 (causal criticality peaks).

  5.5 Early Decision Crystallization

  Prediction: For a significant fraction of generated tokens (especially function words and high-frequency content words), the model's top-1
   prediction is already correct at layer $\ell \ll L$, meaning the remaining layers perform redundant computation for these tokens. The
  crystallization layer distribution is bimodal: easy tokens crystallize very early, hard tokens crystallize very late.

  Significance: Direct implications for early-exit inference, speculative decoding, and understanding the computational budget allocation of
   transformers.

  Evidence path: E2 (crystallization layer analysis) + E1 (late-layer update norms for early-crystallized tokens).

  5.6 Memory Provenance Structure

  Prediction: The model's "consultation" of prior tokens (as measured by hidden-state similarity) follows a structured pattern that reflects
   linguistic dependencies — not just recency. Specifically, at intermediate layers, current tokens are most similar to their syntactic
  heads (not their immediate predecessors), and at late layers, they are most similar to semantically related antecedents.

  Significance: Would establish that activation similarity is a proxy for linguistic dependency structure, connecting activation geometry to
   parsing and interpretation.

  Evidence path: E0 (top-k reference position analysis) + E8 (token-type stratified reference patterns).

  5.7 Task-Specific Circuit Signatures

  Prediction: Different task types (factual recall, mathematical reasoning, narrative generation, code completion) activate different
  subsets of the residual computation, detectable as task-specific profiles in the update magnitude, spectral, and logit-lens measurements.
  These profiles are consistent across different prompts within the same task type.

  Significance: Would demonstrate that the model routes different computations through different layer subsets despite uniform architecture,
   analogous to task-specific cortical activation patterns in neuroscience.

  Evidence path: E1 (cross-dataset update profiles) + E3 (cross-dataset geometry) + E6 (task-conditional causal importance).

  ---
  6. SYSTEM ARCHITECTURE FOR EXPERIMENTS

  6.1 Current Architecture

  The existing codebase follows a modular four-layer pattern per experiment:

  run_*.py  (entry point)
    ├── config/*.yaml          (experiment parameters)
    ├── src/*_experiment.py    (core measurement logic)
    ├── src/*_sweep.py         (parameter sweep orchestration)
    ├── src/*_storage.py       (Parquet I/O + checkpointing)
    └── src/*_analysis.py      (visualization + statistical analysis)

  Shared infrastructure:
  src/model.py       → model loading + hidden-state extraction
  src/metrics.py     → core similarity metrics (GPU-vectorized)
  src/prompts.py     → dataset loading + prompt generation
  src/scheduler.py   → GPU detection + job scheduling
  src/storage.py     → generic Parquet storage

  6.2 Proposed Unified Architecture

  To scale from 6 experiments to the full framework, we need a more systematic architecture:

  activation_science/
  │
  ├── core/                              # Shared infrastructure
  │   ├── model.py                       # Model loading, hidden-state extraction, KV-cache management
  │   ├── extraction.py                  # Unified activation extraction pipeline
  │   │     - extract_all_layers(model, input_ids) → Dict[layer, Tensor]
  │   │     - extract_residual_updates(model, input_ids) → Dict[layer, Tensor]
  │   │     - extract_logit_lens(model, input_ids) → Dict[layer, Tensor]
  │   ├── datasets.py                    # Dataset registry, prompt generation, token annotation
  │   ├── scheduler.py                   # GPU scheduling, job distribution
  │   ├── storage.py                     # Unified Parquet storage with schema registry
  │   └── types.py                       # Shared type definitions (ActivationBatch, ExperimentConfig, etc.)
  │
  ├── metrics/                           # Modular metric library
  │   ├── similarity.py                  # Cosine similarity, CKA, Procrustes, CCA
  │   ├── spectral.py                    # SVD, effective rank, participation ratio, anisotropy
  │   ├── information.py                 # Entropy, KL divergence, mutual information
  │   ├── dynamics.py                    # Velocity, curvature, Lyapunov, recurrence
  │   ├── intervention.py               # Ablation, patching, interchange
  │   └── clustering.py                  # K-means, HDBSCAN, regime detection
  │
  ├── experiments/                       # One module per experiment
  │   ├── base.py                        # Abstract BaseExperiment class
  │   ├── e0_decode_similarity.py
  │   ├── e1_residual_decomposition.py
  │   ├── e2_logit_lens.py
  │   ├── e3_representation_geometry.py
  │   ├── e4_cross_sequence_alignment.py
  │   ├── e5_state_detection.py
  │   ├── e6_causal_intervention.py
  │   ├── e7_perturbation_sensitivity.py
  │   └── e8_token_type_analysis.py
  │
  ├── analysis/                          # Post-hoc analysis and visualization
  │   ├── base.py                        # Abstract BaseAnalysis class
  │   ├── per_experiment/                # Experiment-specific plots
  │   │   ├── e0_plots.py
  │   │   ├── e1_plots.py
  │   │   └── ...
  │   ├── cross_experiment/              # Cross-experiment integration
  │   │   ├── phase_boundary_detection.py
  │   │   ├── semantic_bottleneck_analysis.py
  │   │   ├── regime_taxonomy.py
  │   │   └── circuit_signature_analysis.py
  │   └── statistical.py                 # Hypothesis testing, effect sizes, confidence intervals
  │
  ├── config/                            # YAML configurations
  │   ├── base.yaml                      # Shared defaults (model, GPU, datasets)
  │   ├── experiments/                   # Per-experiment overrides
  │   │   ├── e0.yaml
  │   │   └── ...
  │   └── sweeps/                        # Sweep-specific configurations
  │
  ├── runners/                           # Entry points
  │   ├── run_experiment.py              # Unified runner: `python -m runners.run_experiment e0`
  │   ├── run_analysis.py                # Unified analysis: `python -m runners.run_analysis e0`
  │   └── run_cross_analysis.py          # Cross-experiment analysis
  │
  └── results/                           # Output (gitignored)
      ├── e0_decode_similarity/
      ├── e1_residual_decomposition/
      └── ...

  6.3 Key Design Principles

  Principle 1: Single extraction, multiple metrics. Many experiments need the same hidden states. The extraction pipeline should compute
  hidden states once and feed them to multiple metric functions. Experiments declare their extraction requirements (which layers, which
  phases, whether residual updates are needed), and the extraction layer batches and deduplicates.

  Principle 2: Schema-first storage. Every experiment declares its output schema (column names, types, descriptions) as a dataclass or typed
   dictionary. The storage layer validates writes against the schema and enforces consistency. This prevents schema drift as experiments
  evolve.

  Principle 3: Metric composability. Metrics are pure functions: Tensor → Tensor or Tensor × Tensor → scalar. They can be composed (e.g.,
  "compute SVD, then effective rank, then plot across layers") without side effects. This enables reuse across experiments and makes testing
   straightforward.

  Principle 4: Cross-experiment analysis as first-class citizen. The most important scientific insights will come from combining results
  across experiments (e.g., "phase boundaries from E1 coincide with crystallization layers from E2 and causal importance peaks from E6").
  The architecture should make cross-experiment queries natural, not afterthoughts.

  Principle 5: Reproducibility. Every experiment run produces a deterministic ID from its full parameter set (already implemented via SHA256
   hashing). Results are append-only Parquet files. Configuration is fully specified in YAML. The combination of deterministic IDs +
  append-only storage + version-controlled configs ensures that any result can be reproduced or audited.

  ---
  7. LONG-TERM RESEARCH ROADMAP

  Stage 1: Descriptive Activation Science (Current → +6 months)

  Goal: Establish the empirical foundation by completing all descriptive experiments and building a comprehensive atlas of activation
  behavior across layers, tokens, and datasets.

  Work program:

  ┌──────────┬──────────────────────────────┬─────────────────────┬──────────────────────────────────────────┐
  │ Priority │          Experiment          │       Status        │                Rationale                 │
  ├──────────┼──────────────────────────────┼─────────────────────┼──────────────────────────────────────────┤
  │ ✓ Done   │ E0: Decode Similarity        │ Complete (168 jobs) │ Baseline temporal dynamics               │
  ├──────────┼──────────────────────────────┼─────────────────────┼──────────────────────────────────────────┤
  │ High     │ E1: Residual Decomposition   │ Framework ready     │ Defines layer functional roles           │
  ├──────────┼──────────────────────────────┼─────────────────────┼──────────────────────────────────────────┤
  │ High     │ E2: Logit Lens               │ Framework ready     │ Maps decision formation                  │
  ├──────────┼──────────────────────────────┼─────────────────────┼──────────────────────────────────────────┤
  │ High     │ E3: Representation Geometry  │ Framework ready     │ Characterizes activation space structure │
  ├──────────┼──────────────────────────────┼─────────────────────┼──────────────────────────────────────────┤
  │ Medium   │ E4: Cross-Sequence Alignment │ Framework ready     │ Tests universality hypothesis            │
  ├──────────┼──────────────────────────────┼─────────────────────┼──────────────────────────────────────────┤
  │ Medium   │ E5: State Detection          │ Framework ready     │ Identifies generation regimes            │
  ├──────────┼──────────────────────────────┼─────────────────────┼──────────────────────────────────────────┤
  │ Medium   │ E8: Token-Type Analysis      │ Design complete     │ Validates stratification hypothesis      │
  └──────────┴──────────────────────────────┴─────────────────────┴──────────────────────────────────────────┘

  Execution order rationale: E1–E3 are independent and characterize the three most fundamental aspects (computation, decisions, geometry).
  They should run first, in parallel if GPU resources allow. E4–E5 build on their findings. E8 can run in parallel with E4–E5.

  Deliverables:
  - Complete measurement dataset across all descriptive experiments
  - Per-experiment analysis reports with confirmed/refuted hypotheses
  - Cross-experiment integration: phase boundary analysis, semantic bottleneck localization
  - Publication-quality figures for each major finding
  - Open dataset release (Parquet files + metadata)

  Key scientific questions to answer:
  1. Do layer phase transitions exist, and where are they?
  2. Is there a semantic bottleneck, and which layers define it?
  3. How many generation regimes exist, and what characterizes each?
  4. Does early decision crystallization occur, and for which token types?

  ---
  Stage 2: Causal Activation Experiments (+6 → +12 months)

  Goal: Move from description to causation. Use interventional experiments to determine which activation properties are causally necessary
  for model behavior, not merely correlated.

  Work program:

  ┌──────────┬──────────────────────────────┬──────────────────────────┬────────────────────────────────────────────────────────────────┐
  │ Priority │          Experiment          │       Dependencies       │                           Rationale                            │
  ├──────────┼──────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ High     │ E6: Causal Layer             │ E1 (phase boundaries)    │ Tests whether identified phases are causally meaningful        │
  │          │ Intervention                 │                          │                                                                │
  ├──────────┼──────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ High     │ E7: Perturbation Sensitivity │ E3 (geometry)            │ Tests whether geometric structure reflects functional          │
  │          │                              │                          │ importance                                                     │
  ├──────────┼──────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Medium   │ Targeted ablation studies    │ E6 (critical layers)     │ Zooms in on critical components                                │
  ├──────────┼──────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Medium   │ Cross-model validation       │ E0–E5 (descriptive       │ Tests universality of findings across model families           │
  │          │                              │ atlas)                   │                                                                │
  └──────────┴──────────────────────────────┴──────────────────────────┴────────────────────────────────────────────────────────────────┘

  New methods required:
  - Activation patching infrastructure (intercept forward pass, replace activations)
  - Efficient perturbation sweep (many directions × many layers × many tokens)
  - Cross-model extraction (support for Llama, Mistral, GPT-NeoX architectures)

  Key scientific questions to answer:
  1. Are the layer phases identified in Stage 1 causally distinct (ablating a phase boundary layer causes different effects than ablating a
  mid-phase layer)?
  2. Do high-sensitivity directions in activation space align with the principal directions identified by SVD?
  3. Are the findings from Qwen2.5-32B universal across model families, or are they architecture-specific?

  ---
  Stage 3: Mechanistic Circuit Discovery (+12 → +24 months)

  Goal: Connect activation-level findings to specific computational circuits — attention heads, MLP neurons, and their interactions that
  implement the observed activation dynamics.

  Work program:

  ┌──────────┬────────────────────────────────┬─────────────────────────────────────────┬──────────────────────────────────────────────┐
  │ Priority │           Direction            │              Dependencies               │                  Rationale                   │
  ├──────────┼────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ High     │ Phase-boundary circuit         │ E1, E6 (phase boundaries + causal       │ What circuits create phase transitions?      │
  │          │ identification                 │ importance)                             │                                              │
  ├──────────┼────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ High     │ Decision crystallization       │ E2, E6 (crystallization layers + causal │ What circuits "commit" to a prediction?      │
  │          │ circuits                       │  ablation)                              │                                              │
  ├──────────┼────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Medium   │ Regime transition mechanisms   │ E5 (regimes) + E6 (interventions)       │ What triggers generation regime shifts?      │
  ├──────────┼────────────────────────────────┼─────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Medium   │ Memory routing circuits        │ E0 (reference patterns) + attention     │ How does the model "choose" which prior      │
  │          │                                │ analysis                                │ tokens to consult?                           │
  └──────────┴────────────────────────────────┴─────────────────────────────────────────┴──────────────────────────────────────────────┘

  New methods required:
  - Attention head attribution (which heads contribute to observed activation patterns)
  - MLP neuron analysis (which neurons activate during specific phenomena)
  - Circuit grafting (transplanting circuits between models to test sufficiency)
  - Automated circuit discovery (scalable methods beyond manual analysis)

  Key scientific questions to answer:
  1. Are phase transitions caused by specific attention heads or by emergent properties of the full layer?
  2. Do decision crystallization circuits generalize across token types, or are there separate circuits for different linguistic categories?
  3. Can generation regimes be controlled by intervening on specific circuit components?

  ---
  Stage 4: Predictive Theory of LLM Computation (+24 → +36 months)

  Goal: Synthesize findings into a quantitative theory that predicts activation behavior from model architecture and training data, without
  requiring empirical measurement of each new model.

  Theoretical targets:

  1. Phase transition theory: A model of how network depth, width, and training data determine the number and location of computational
  phases. Predict where phase boundaries will occur in a new model from its architecture alone.
  2. Decision formation dynamics: A dynamical model of how prediction entropy decreases across layers as a function of token predictability
  and context length. Predict the crystallization layer distribution for a new model on a new dataset.
  3. Representational capacity theory: A theory relating model dimension, depth, and training distribution to the effective dimensionality
  and anisotropy profile of activation space. Predict geometric properties from architecture.
  4. Generation regime theory: A dynamical systems model of autoregressive generation that predicts the number, character, and transition
  probabilities of generation regimes from model properties.

  Validation methodology: Train or obtain access to model families of varying sizes (1B, 7B, 13B, 32B, 70B) and test whether theoretical
  predictions hold across scales. Discrepancies between prediction and observation drive theory refinement.

  Ultimate deliverable: A textbook-quality theoretical framework — "Activation Science of Large Language Models" — that provides the same
  role for LLM interpretability that statistical mechanics provides for thermodynamics: a microscopic theory (circuits, attention, neurons)
  connected to macroscopic observables (activation geometry, decision dynamics, generation regimes) through principled analytical tools.

  ---
  Appendix A: Notation Reference

  ┌───────────────────┬────────────────────────────────────────────────────────┐
  │      Symbol       │                        Meaning                         │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $h_\ell^{(t)}$    │ Hidden state at layer $\ell$, token position $t$       │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $\Delta_\ell$     │ Residual update at layer $\ell$: $h_{\ell+1} - h_\ell$ │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $f_\ell$          │ Combined attention + MLP function at layer $\ell$      │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $W_U$             │ Unembedding matrix (hidden → vocabulary logits)        │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $\sigma_i$        │ $i$-th singular value of a hidden-state matrix         │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $d$               │ Hidden dimension of the model                          │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $L$               │ Total number of layers                                 │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $T$               │ Sequence length (number of tokens)                     │
  ├───────────────────┼────────────────────────────────────────────────────────┤
  │ $\text{erank}(H)$ │ Effective rank (exponential of spectral entropy)       │
  └───────────────────┴────────────────────────────────────────────────────────┘

  Appendix B: Experiment Dependency Graph

  E0 (decode similarity)  ─────────────────────────────────────┐
      ↓                                                         │
  E1 (residual decomp) ──→ E6 (causal intervention) ──→ Circuit │
      ↓                         ↑                      Discovery│
  E2 (logit lens) ─────────────┘                               │
      ↓                                                         │
  E3 (geometry) ────────→ E7 (perturbation sensitivity) ───────┤
      ↓                                                         │
  E4 (cross-sequence) ─→ Semantic Bottleneck Analysis ─────────┤
      ↓                                                         │
  E5 (state detection) ─→ Regime Taxonomy ─────────────────────┤
      ↓                                                         │
  E8 (token-type) ──────→ Stratification Analysis ─────────────┘
                                                                │
                                                Predictive Theory ←──┘

  Appendix C: Existing Infrastructure Mapping

  ┌───────────────────────────────┬──────────────────────────────────────┬─────────────────────────────┐
  │      Framework Component      │        Current Implementation        │           Status            │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Model loading                 │ src/model.py                         │ Production                  │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Hidden-state extraction       │ src/model.py:extract_hidden_states() │ Production                  │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Cosine similarity metrics     │ src/metrics.py                       │ Production (GPU-vectorized) │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Dataset loading + prompts     │ src/prompts.py                       │ Production (7 datasets)     │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ GPU scheduling                │ src/scheduler.py                     │ Production (pynvml-based)   │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Parquet storage               │ src/storage.py + per-experiment      │ Production                  │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Checkpointing                 │ Per-experiment *_storage.py          │ Production (SHA256 IDs)     │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Decode similarity analysis    │ src/decode_analysis.py (27 plots)    │ Production                  │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Residual decomposition        │ src/residual_experiment.py           │ Ready (not run)             │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Logit lens                    │ src/logit_lens_experiment.py         │ Ready (not run)             │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Representation geometry       │ src/geometry_experiment.py           │ Ready (not run)             │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Cross-sequence alignment      │ src/cross_sequence_experiment.py     │ Ready (not run)             │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ State detection               │ src/state_experiment.py              │ Ready (not run)             │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Causal intervention (E6)      │ Not implemented                      │ Proposed                    │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Perturbation sensitivity (E7) │ Not implemented                      │ Proposed                    │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Token-type analysis (E8)      │ Not implemented                      │ Proposed                    │
  ├───────────────────────────────┼──────────────────────────────────────┼─────────────────────────────┤
  │ Cross-experiment analysis     │ Not implemented                      │ Proposed                    │
  └───────────────────────────────┴──────────────────────────────────────┴─────────────────────────────┘

  ---
  This document is the foundational blueprint for the Activation Science research program. Each section is designed to be independently
  actionable while contributing to the coherent whole. The framework transforms ad-hoc experiments into a systematic investigation of how
  large language models represent, transform, and use information during computation.