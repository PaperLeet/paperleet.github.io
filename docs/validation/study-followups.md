# Deeper self-study prompts and metadata discovery

Thirty bilingual follow-up questions extend fifteen maintained study guides:

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
all thirty questions, malformed/duplicate/oversized and legacy content shapes,
independent reveal/collapse/reopen, language/topic changes, notes locking and
metadata-based discovery. The existing full handbook fixture checks all 89/267
identities, curricula, stale imports/timeouts, practice/sign-in context, search
and private editor/selection/scroll/storage continuity. Production builds check
initial asset budgets and static route integrity. No physical browser layout,
numerical SDK execution, service mutation or deployment is claimed.

Each content expansion adds ten follow-ups in five previously unenriched guides.
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
