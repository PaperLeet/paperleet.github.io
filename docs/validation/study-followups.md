# Deeper self-study prompts and metadata discovery

Eighty-six bilingual follow-up questions extend forty-three maintained study guides:

| Exercise | Follow-up topics | Statement basis |
| --- | --- | --- |
| architecture/attention | Top-left causal alignment with cached queries; size of returned weights | README formula/conventions, interface and toy scope |
| architecture/rope | Adjacent-coordinate versus half pairing; per-example positions outside the current interface | README formula-to-code, contract and extension |
| architecture/rms_norm | Prefix statistics under a feature permutation; token independence | README formulas, prefix selection and per-token scope |
| finetuning/lora | Factorized forward intermediates; accidental double application after merging | README formula, adapter shapes, merge and input-preservation contract |
| rl/dpo | Sequence sums versus means; shared score offsets and diagnostic reward shifts | README objective, sequence aggregation and reference-adjusted rewards |
| architecture/gqa | Cached-query mask alignment; persistent KV capacity versus temporary computation storage | README explicit mask alignment and cache-count scope |
| architecture/multihead_attention | Non-symmetric projection orientation; one masked head versus dense output mixing | README x @ w convention, per-head masking and bias-free output projection |
| architecture/layer_norm | Positive scaling with epsilon; independent residual-branch gradients | README normalization formula, scaling identity and post-LN interface |
| architecture/positionwise_ffn | Closed ReLU gates with surviving output bias; token permutation versus shared parameters | README two affine layers, ReLU and position independence |
| optimization/adamw | Zero current gradient with nonzero history; raw state under constant gradients | README moment recurrence, one-based bias correction and functional state contract |
| rl/ppo_clip | Zero surrogate loss with live log-prob gradients; zero clipping width versus freezing | README pessimistic minimum, negative batch mean and frozen targets |
| rl/gae | Rollout partition carry; explicit next-value consistency for telescoping | README zero incoming carry, two ending masks and per-step next_values |
| rl/grpo | Reward units with fixed epsilon; empty response groups versus token diagnostics | README sample std plus epsilon and distinct response/token denominators |
| rl/dr_grpo | Unequal microbatch aggregation; reward scale versus retained KL | README fixed N*max_len denominator, token diagnostics and centered-only advantages |
| architecture/flash_attention | Disjoint-state merge; row shifts change LSE but preserve output | README online unnormalized states, final u/l and m+log(l), retained LSE gradients |

| foundations/next_token_loss | Vocabulary permutation with inverse target mapping; row-wise shift and zero-sum logit gradients | README shifted NLL, stable vocabulary normalization and batch scope |
| foundations/sft_loss_mask | Prompt context versus direct logit gradients; first-answer predecessor counterexample | README target-only response/attention gates and one internal shift |
| foundations/gradient_accumulation | Hierarchical mean/count merging; optimizer-update boundary | README local token means, fixed supplied gradients and pure merge scope |
| foundations/gradient_clipping | Parameter partition invariance; clipping/weighted-merge noncommutation | README global L2 norm, shared epsilon scale and no mutation |
| foundations/top_p_sampling | Nested prefixes and renormalization; top-k pretruncation counterexample | README full-vocabulary shortest prefix, temperature/tie rules and final normalization |

| representation/clip | Paired versus one-sided permutations; single-pair temperature-gradient degeneracy | README diagonal targets, two retrieval directions and clipped log scale |
| representation/simclr | Candidate membership versus loss factors; equal-score candidate baseline | README self exclusion, same-view negatives and 2B-anchor mean |
| representation/moco | Chunk-composed circular enqueue; EMA with changing online parameters | README sequential overwrites, returned pointer, detached functional EMA |
| representation/vicreg | Duplicated-batch sample statistics; missing cross-group covariance | README B-1 statistics, elementwise MSE and both-view off-diagonal sum |
| representation/ijepa | Repeated-mask occurrence weights; smooth-L1 scaling regimes | README ordered gather, all-element mean, beta=1 and stopped targets |

| retrieval/bpr | Common item shifts versus the full-table prior; unequal microbatch prior counting | README pairwise data mean, full-table L2 once and empty-batch scope |
| retrieval/sasrec | Common logit shifts change a binary objective; per-target versus per-sequence weighting | README independent binary terms, target mask and valid-position denominator |
| retrieval/lightgcn | Symmetric normalization versus row means; bipartite layer-parity cancellation | README symmetric adjacency, no self-loops and layer-zero uniform averaging |
| representation/graphsage_mean | Sampling multiplicity and feature derivatives; weight scaling near the norm floor | README sampled slots, repeated IDs, ReLU and per-hop epsilon normalization |
| retrieval/late_interaction | Asymmetric token duplication; score monotonicity under document extension | README valid-token max, query sum, first-valid tie gradients and masked positions |
| retrieval/semantic_ids | Greedy versus jointly optimal reconstruction; chunk-local collision counters | README frozen residual encoding, lowest-index ties and complete-tuple occurrence order |
| retrieval/generative_retrieval | Prefix-local logit-shift invariance; catalog-only renormalization can reverse order | README full-vocabulary log-softmax, global beam pruning and original-score lookup |

| foundations/top_k_sampling | Large-temperature retained support; selected ranks versus floating-point nonzero probabilities | README exact-k ranking, finite temperature, float64 intermediates and final dtype |
| foundations/top_p_draw | Vocabulary permutation versus fixed-draw coupling; finished-row RNG policy outside the single-step interface | README vocabulary-order CDF, per-row uniform contract and EOS scope |
| foundations/multiturn_sft_mask | Single-shift loss bridge; valid-target-weighted microbatch means | README target-position metadata and existing SFT loss conventions |
| finetuning/ewc | Independent historical snapshots; Fisher-weighted quadratic merging and its additive constant | README fixed snapshots, single-task penalty and original EWC objective; PyTorch detach storage documentation |
| foundations/continual_metrics | Initialization baseline versus learned-task reference; equal-task versus pooled-example accuracy | README/GEM matrix protocol and equal-task reduction |

| architecture/positional_encoding | Paired dot products and odd-width exceptions; computable positions versus precision and model extrapolation | README paired sine/cosine formula, odd-width extension, explicit dtype and toy scope |
| architecture/group_norm | Group-shift gradient direction; spatial/channel permutations and retained group membership | README sample/group moments, channel affine parameters and stateless interface |
| finetuning/qlora | Transposition changes flattened blocks; additional error after adapter merge and requantization | README row-major absmax blocks, frozen reconstruction and additive adapter |
| architecture/residual_block | First-step gradients with zero W2; identity-valued trainable shortcut versus parameter-free identity | README post-addition ReLU, right-multiplied weights and zero-second-layer toy initialization |
| architecture/batch_norm | Microbatch/EMA composition; fixed inference folding into a preceding linear map | README batch/spatial moments, new-batch momentum convention and detached inference buffers |
| representation/vae | Zero noise changes the estimator and gradient paths; expected ELBO bound versus a single draw | README caller-supplied noise, analytic KL, single-sample likelihood term and toy limits |

These are editorial discussions based on the repository exercise statements.
They are not presented as reported interview questions, frequency counts or a
new assessment. All 89 guides and 267 existing core prompts retain their source
and function identities. Full-notes unlocking and numerical grades are unchanged.

The optional follow-up collection accepts one to four unique bounded IDs with
complete bounded Chinese/English questions and hints. The existing deferred JSON
import carries this content. An unopened guide does not request it. Follow-ups
are a native collapsed disclosure with individually hidden hints; settled
collapse/reopen and locale changes preserve native reading state. The existing
local Collapse all hints action covers both prompt groups and identifies both
controlled regions. Another exercise retires old follow-up nodes and disclosure
state; guides without additional content offer no empty section.

Handbook search now includes loaded bilingual implementation goals and paper
titles alongside names, numbers and functions. It keeps literal normalized AND
matching and the current reader until an explicit choice. It does not load hint
content, change a report query, or write a grade to answer a search.

Checks cover the actual unmocked deferred content loader and StrictMode reader,
all eighty-six questions, malformed/duplicate/oversized and legacy content shapes,
independent reveal/collapse/reopen, language/topic changes, notes locking and
metadata-based discovery. The existing full handbook fixture checks all 89/267
identities, curricula, stale imports/timeouts, practice/sign-in context, search
and private editor/selection/scroll/storage continuity. Production builds check
initial asset budgets and static route integrity. No physical browser layout,
numerical SDK execution, service mutation or deployment is claimed.

The earlier expansions each added ten follow-ups in five previously unenriched guides.
Previous follow-ups, core prompts, source paths and function identities remain
byte-equivalent as parsed objects. Both statement languages were reviewed.
No numerical run is needed or claimed for these editorial discussion prompts.

Learning-state and criteria-empty concept results now retain the same result
region ID that the sort control identifies. The existing Show all learning states
action restores visible rows; the relationship remains valid while no rows match.
An actual StrictMode regression first reproduced the missing target and checks
its recovery without changing filters, chronology or private editor data.

The reinforcement-learning/online-softmax expansion adds ten discussions in five
previously unenriched guides. Content review compares all 89 parsed guides to the
published parent, requires every prior field and twenty earlier follow-ups to be
identical, and records both statement-language hashes. The new questions derive
from the local contracts: sample std and epsilon, fixed old/reference targets,
response versus valid-token denominators, explicit GAE next-values and finite
unmasked attention scores. They do not add exercises or numerical grading rules.

The supporting original papers were consulted as background: [PPO](https://arxiv.org/pdf/1707.06347),
[GAE](https://arxiv.org/html/1506.02438v6#S3),
[GRPO](https://arxiv.org/html/2402.03300v3#S4.SS1),
[Dr. GRPO](https://arxiv.org/html/2503.20783v2#S3.SS2), and
[FlashAttention](https://arxiv.org/html/2205.14135).
The added counterexamples and algebra are editorial derivations from PaperLeet's
stated interfaces, including its retained-KL teaching variant and zero incoming
GAE carry, rather than claims of matching every paper implementation. No
training run, GPU performance, true interview frequency or optimizer-scale
invariance is inferred from these discussions.

The training-foundations expansion adds ten editorial discussions in five guides.
The published parent retains all 89 identities, all 267 core prompts and the prior
30 follow-ups. Review records both README-language hashes and compares every
parsed parent field, allowing only the ten explicitly named follow-ups. The
three-microbatch, three-token and top-k/top-p examples are hand-derived from the
local contracts; they are not new execution tests, interview reports, optimizer
benchmarks or claims about full-model training equivalence.

Background references were checked against [causal language modeling](https://huggingface.co/docs/transformers/main/en/tasks/language_modeling),
[assistant-only SFT loss](https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-assistant-messages-only),
[PyTorch accumulation](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html#gradient-accumulation),
[global norm clipping](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.clip_grad_norm_.html),
and the [nucleus-sampling paper](https://arxiv.org/html/1904.09751v2#S3.SS1).
PaperLeet's pure-function/valid-token interfaces and explicit shortest-prefix
boundary remain authoritative; these are not literal reproductions of trainer,
in-place optimizer utility or released sampler implementations.


The representation-path expansion adds ten discussions across all five modules of
that maintained path. The previous 40 follow-ups, 267 core prompts and 89 guide
identities remain unchanged as parsed objects. Counterexamples and formulas are
editorial derivations from the local exercise contracts. I-JEPA's supplied
validator accepts repeated valid indices; this does not assert that the original
training mask generator produces duplicates. No encoder training, numerical
judge execution, new grading rule or reported interview frequency is claimed.

Primary background includes the original [CLIP](https://proceedings.mlr.press/v139/radford21a.html)
and [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) papers,
[MoCo](https://arxiv.org/html/1911.05722v3), and the pinned author code for
[VICReg](https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py)
and [I-JEPA](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py).
The maintained README contract controls each exercise's implementation and
normalization conventions; the added algebra is not a claim of reproducing
paper benchmarks or every original training detail.


The recommendation/retrieval expansion covers all seven modules in the existing
maintained path, adding14 discussions. The previous50 follow-ups,267 core prompts
and89 identities remain equal as parsed objects. Examples are editorial algebra
from the exercise contracts: fixed-factor prior aggregation, valid-target means,
symmetric graph weights and alternating propagation, sampled-slot derivatives,
epsilon regimes, MaxSim reductions, greedy quantization and frozen decoding.
They introduce no new runtime, grading, training result or interview-frequency
claim. Streaming collision state and catalog-normalized decoding are discussed
as changes to the current contract, not implemented as hidden extensions.

Primary background includes [BPR](https://proceedings.mlr.press/r7/rendle09a.html),
the pinned [SASRec objective](https://github.com/kang205/SASRec/blob/e3738967fddab206d6eeb4fda433e7a7034dd8b1/model.py),
[LightGCN](https://arxiv.org/html/2002.02126),
[GraphSAGE](https://arxiv.org/html/1706.02216),
[ColBERT](https://arxiv.org/html/2004.12832), and
[TIGER](https://arxiv.org/html/2305.05065). Local README interfaces control
normalizations, tie rules, epsilon behavior and teaching boundaries.


The generation/continual-learning expansion adds ten discussions in five formerly
unenriched modules, completing follow-up coverage for the maintained
`generation-continual-basics` path. All64 earlier follow-ups,267 core prompts and89
guide identities remain unchanged as parsed objects. The limit, finite-precision,
CDF-coupling, single-shift, weighted-mean and quadratic-completion discussions are
editorial derivations from the local contracts. EOS orchestration, snapshot
management, historical-task consolidation and alternative metric protocols are
explicit extensions for discussion; this wave does not implement them or change
grading, sampling streams, account data or numerical runtime behavior.

Background was checked against [Transformers generation configuration](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation),
[the nucleus-sampling paper](https://arxiv.org/abs/1904.09751),
[TRL assistant-only loss and label shifting](https://huggingface.co/docs/trl/sft_trainer),
[the EWC objective](https://arxiv.org/html/1612.00796v2#S2.E3),
[PyTorch detach storage semantics](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html),
and [GEM evaluation metrics](https://papers.neurips.cc/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).
The exact-k ties, original-vocabulary inverse CDF, target-mask alignment and
nonnegative peak-forgetting convention remain PaperLeet-specific contracts.
No new numerical execution, training result or reported interview frequency is
claimed; the actual deferred reader checks both languages for every added item.

The architecture/vision-foundations expansion adds twelve discussions in six
previously unenriched guides. All74 prior follow-ups,267 core prompts and89 guide
identities remain identical as parsed objects. Both statement languages were
reviewed. Every module in the maintained `transformer`, `normalization`, `llm`
and `vision-foundations` paths now has follow-ups. The existing reader checks
every new question and hint in both languages, while an unrelated unenriched
guide continues to cover the absent-content state.

Primary background includes [Transformer positional encoding](https://arxiv.org/html/1706.03762v7#S3.SS5),
[Group Normalization](https://arxiv.org/html/1803.08494v3#S3),
[QLoRA](https://arxiv.org/html/2305.14314v1),
[ResNet shortcuts](https://arxiv.org/html/1512.03385v1#S3.SS2),
[BatchNorm training and inference](https://arxiv.org/html/1502.03167v3#S3),
and [the VAE estimator](https://arxiv.org/html/1312.6114v11#S3).
The paired-dot, permutation, block-layout, gradient, EMA and single-draw examples
are editorial derivations from PaperLeet's local contracts. In particular,
odd-width positional encoding is a platform extension; full-model long-context
quality is outside its toy checks. The group-shift gradient statement applies to loss dependence through that GN
output, not unrelated input paths. A zero-noise decoder and a stochastic ELBO
estimate are distinguished from a posterior expectation. The likelihood-bound
interpretation requires observations in the generative model’s sample space;
accepted soft targets do not become binary Bernoulli events. Inference folding and
requantized adapter deployment remain discussion topics, not newly implemented
interfaces. No numerical run, packed-memory benchmark, training claim, grading
change or reported interview frequency is introduced.
