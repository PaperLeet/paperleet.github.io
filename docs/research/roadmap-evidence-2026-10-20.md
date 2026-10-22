# Collection roadmap evidence · 2026-10-20

Each featured collection roadmap shows only papers whose importance is backed by at least two independent kinds of evidence: surveys, citation data, course syllabi, expert or community discussion, and adoption in later papers, libraries and model releases. Each paper's metadata and inclusion, the completeness of each candidate list, and every arrow are checked against the cited sources. The mixture-of-experts papers in LLM Frontier (GShard, Switch Transformer, DeepSeekMoE) meet the same evidence rule; their entries record arXiv and venue metadata as separate items.

The curated roadmaps apply the site rules. High-confidence papers come first, then papers that start or connect a line. A map has at most 14 papers. Every roadmap paper has an arXiv version, so the panel can show its arXiv card; the seven research core papers without one (REINFORCE, the Policy Gradient Theorem, Netflix MF, FPMC, DSSM, YouTube DNN and Path to AMI) are left out. A paper has at most two incoming arrows, no arrow skips an existing path, and the layout has no crossing or pass-through arrows. Research core papers that are not on the roadmap are named under each roadmap, with the reason in its curation notes.

Arrows mean "the later paper builds directly on the earlier one". Every arrow carries a one-line relation sentence (shown in both papers' panels), written from the papers with a source and a quote and then fact-checked against them. Years are first public release (arXiv v1 or first publication). Evidence notes below are shortened; the [structured record](roadmap-evidence-2026-10-20.json) keeps them in full. Figures and sources marked "(not re-fetched)" were collected once, during candidate research, and not fetched again; "(re-fetched)" and "(fetched 2026-10-20)" mark figures that were fetched again for the check; figures with no marker were collected on 2026-10-20 during candidate research. Confidence is rated per roadmap, so a paper can carry different levels in two roadmaps. Arrow and paper rationales from the research ("why" and "research_why_en" in the structured record) are research-stage text; the roadmap sentence is the curated one shown on the site. Reasons for borderline, rejected and off-roadmap papers are judged against the full candidate pool, so they can mention papers that are not on the roadmap. The curated roadmap and its curation notes are authoritative.

## LLM Frontier

How the 2017 Transformer became the modern LLM stack: decoder-only scaling (GPT-3), the LLaMA recipe and its parts (RMSNorm, SwiGLU, RoPE), efficient attention (FlashAttention, GQA), and a separate mixture-of-experts path from Sparse MoE through GShard, Switch Transformer and DeepSeekMoE to the DeepSeek models.

| Paper | Year | Venue | Exercise | Confidence |
| --- | --- | --- | --- | --- |
| [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | 2017 | ICLR 2017 | grey | high |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | NeurIPS 2017 | `transformer` | high |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 2019 | NeurIPS 2019 | `rmsnorm` | high |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 2020 | arXiv | grey | high |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | 2020 | NeurIPS 2020 | grey | high |
| [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) | 2020 | ICLR 2021 | grey | high |
| [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | 2021 | JMLR 2022 | grey | high |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 2021 | Neurocomputing 2024 | `roformer` | high |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | 2022 | NeurIPS 2022 | `flash-attention` | high |
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 2023 | arXiv | grey | high |
| [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | 2023 | EMNLP 2023 | `gqa` | high |
| [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) | 2024 | ACL 2024 | grey | high |
| [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) | 2024 | arXiv | grey | high |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 2024 | arXiv | grey | high |

**Evidence per paper** (type, linked source, shortened note):

- **Sparse MoE** (2017): Routes each token to a few expert feed-forward networks, growing parameters without growing per-token compute.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1701.06538?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 5,962 citations, 560 influential (not re-fetched).
  - survey · [Cai et al., A Survey on Mixture of Experts in LLMs (arXiv 2407.06204)](https://arxiv.org/html/2407.06204v2): Says sparsely-gated MoE 'has brought new vitality to this three-decade-old technology' and that Shazeer et al. 'pioneered a differentiable heuristic with …
  - community · [Hugging Face blog, Mixture of Experts Explained (Dec 2023)](https://huggingface.co/blog/moe): Its history section presents Shazeer et al. 2017 as the step that scaled MoE to 137B parameters through sparsity (not re-fetched).
  - adoption · [DeepSeek-V2 technical report](https://arxiv.org/html/2405.04434): Cites Shazeer et al. (2017) on the routing-collapse risk that its expert-level balance loss is designed to limit.
- **Transformer** (2017): Replaced recurrence with self-attention plus feed-forward blocks; the architecture modern LLMs are built on.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03762?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 197,457 citations, 20,997 influential (not re-fetched).
  - community · [Sebastian Raschka, Understanding Large Language Models: A Transformative Reading List](https://www.sebastianraschka.com/blog/2023/llm-reading-list.html): Item 2, under 'Understanding the Main Architecture and Tasks'.
  - syllabus · [Stanford CS336 (Spring 2026) course page](https://cs336.stanford.edu/): In Assignment 1, students implement Transformer components and a minimal training run.
  - adoption · [Kimi K3 technical report](https://arxiv.org/html/2607.24653v1): Cites the Transformer as the step that 'replaced recurrence with attention' and presents Attention Residuals as the same idea applied over depth.
- **RMSNorm** (2019): Normalizes by the root mean square only, dropping mean-centering; used in LLaMA and most later LLMs.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1910.07467?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 2,114 citations, 119 influential (not re-fetched).
  - survey · [Zhao et al., A Survey of Large Language Models (arXiv 2303.18223), Section IV-B](https://arxiv.org/html/2303.18223): 'it is suggested to choose the pre RMSNorm for layer normalization'.
  - syllabus · [Stanford CS336 (Spring 2026) Lecture 3: Architectures (third-party summary)](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-lVynu4/): 'RMS Norm is universally adopted over Layer Norm in modern models.'
  - adoption · [LLaMA paper, Section 2.2](https://arxiv.org/html/2302.13971): Pre-normalization with the RMSNorm function of Zhang and Sennrich (2019). The Kimi K3 report also cites RMSNorm (ref. 145) for its KDA output and LatentMoE.
- **SwiGLU** (2020): Showed gated feed-forward layers like SwiGLU beat ReLU and GELU; now the default FFN in LLMs.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2002.05202?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 2,374 citations, 144 influential, for a 4-page arXiv-only note (not re-fetched).
  - survey · [Zhao et al., A Survey of Large Language Models](https://arxiv.org/html/2303.18223): Recommends 'SwiGLU or GeGLU as the activation function'.
  - syllabus · [Stanford CS336 (Spring 2026) Lecture 3 (third-party summary)](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-lVynu4/): GLU variants 'are now standard in nearly all credible modern language models'.
  - community · [Sebastian Raschka, The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison): 'the more efficient SwiGLU has replaced activation functions like GELU.'
  - adoption · [LLaMA paper, Section 2.2](https://arxiv.org/html/2302.13971): Replaces the ReLU non-linearity with SwiGLU, introduced by Shazeer (2020), following PaLM.
- **GPT-3** (2020): Scaled a decoder-only Transformer to 175B parameters and showed few-shot learning improves with scale.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2005.14165?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 64,864 citations, 5,569 influential (not re-fetched).
  - syllabus · [Stanford CS336 (Spring 2026) Lecture 3 (third-party summary)](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-lVynu4/): GPT-3 is one of the reference models used to discuss block design and activation choices.
  - adoption · [LLaMA paper, Section 2.2](https://arxiv.org/html/2302.13971): Attributes its pre-normalization to GPT-3 ('Pre-normalization [GPT3]') and benchmarks LLaMA-13B against GPT-3 (175B).
  - adoption · [Chinchilla paper, Section 1 and Table 1](https://arxiv.org/pdf/2203.15556): 'Following Kaplan et al. (2020) and the training setup of GPT-3 (Brown et al., 2020), many of the recently trained large models have been trained for …
- **GShard** (2020): Scaled MoE Transformers to 600B parameters with top-2 routing, expert capacity and the balance loss most later MoEs reuse.
  - arxiv · [arXiv 2006.16668](https://arxiv.org/abs/2006.16668): v1 Tue, 30 Jun 2020 10:42:02 UTC, only version. Title: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. Sec 2.2: 'The …
  - venue · [ICLR 2021 virtual site, poster page](https://iclr.cc/virtual/2021/poster/3196): ICLR 2021 poster (the OpenReview page sits behind a bot-verification wall, so the ICLR virtual page is cited).
  - survey · [Cai et al., A Survey on Mixture of Experts in LLMs (arXiv 2407.06204)](https://arxiv.org/abs/2407.06204): Cai et al. survey. 'Simplifying the above design, GShard [25] defines a new differentiable auxiliary loss Laux using a differentiable approximation (the …
  - community · [Hugging Face blog, Mixture of Experts Explained (Dec 2023)](https://huggingface.co/blog/moe): Post dated Dec 11, 2023: 'GShard replaces every other FFN layer with an MoE layer using top-2 gating in both the encoder and the decoder.' On expert capacity …
  - syllabus · [Stanford CS336 (Spring 2025) Lecture 4: MoEs](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf): CS336 Spring 2025 Lecture 4. The routing slide lists 'Gshard (k=2), Grok (2), Mixtral (2)', and GShard is the first row of the routing-setup table.
  - adoption · [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437): DeepSeek-V3 Sec 2.1.2: 'Compared with traditional MoE architectures like GShard (Lepikhin et al., 2021), DeepSeekMoE uses finer-grained experts and isolates …
- **Switch** (2021): Routed each token to just one expert and kept the router in float32 for stability, scaling Transformers to 1.6T parameters.
  - arxiv · [arXiv 2101.03961](https://arxiv.org/abs/2101.03961): v1 Mon, 11 Jan 2021 16:11:52 UTC (v2 30 Apr 2022, v3 16 Jun 2022). Comments field: 'JMLR'. In the paper: 'a simplified strategy where we route to only a single …
  - venue · [JMLR paper page](https://jmlr.org/papers/v23/21-0998.html): JMLR 23(120):1-39, 2022.
  - community · [Hugging Face blog, Mixture of Experts Explained (Dec 2023)](https://huggingface.co/blog/moe): Dedicated section: 'Contrary to the initial idea of using at least two experts, Switch Transformers uses a simplified single-expert strategy.' The post also …
  - survey · [Cai et al., A Survey on Mixture of Experts in LLMs (arXiv 2407.06204)](https://arxiv.org/abs/2407.06204): Cai et al.: 'the Switch Transformer model [34] has shown that a top-1 gating strategy (as illustrated in Figure 4 (a)) can also yield competitive results, a …
  - syllabus · [Stanford CS336 (Spring 2025) Lecture 4: MoEs](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf): CS336 L4 opens 'What's a MoE?' with a [Fedus et al 2022] figure. 'Per-expert balancing – same as the switch transformer'. 'Solution: Use Float 32 just for the …
  - adoption · [DeepSeek-V2 technical report](https://arxiv.org/abs/2405.04434): DeepSeek-V2 Sec 2.2: 'We use an expert-level balance loss (Fedus et al., 2021; Lepikhin et al., 2021) to mitigate the risk of routing collapse'.
- **RoPE** (2021): Encodes position by rotating queries and keys so attention depends on relative offset; used by most LLMs.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2104.09864?fields=title,year,venue,citationCount,influentialCitationCount): 6,864 citations, 561 influential (not re-fetched).
  - survey · [Zhao et al., A Survey of Large Language Models](https://arxiv.org/html/2303.18223): 'RoPE or ALiBi is a better choice since it performs better on long sequences.'
  - syllabus · [Stanford CS336 (Spring 2026) Lecture 3 (third-party summary)](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-lVynu4/): 'RoPE ... has become dominant for position dependence since 2024.'
  - adoption · [LLaMA paper, Section 2.2; DeepSeek-V4 report, Section 2.3.3](https://arxiv.org/html/2302.13971): LLaMA removes absolute positional embeddings and adds RoPE (Su et al., 2021). DeepSeek-V4 still applies partial RoPE, citing Su et al., 2024.
- **FlashAttention** (2022): Computes exact attention in tiles kept in fast on-chip memory, making long sequences far cheaper.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2205.14135?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 5,618 citations, 478 influential (not re-fetched).
  - syllabus · [Stanford CS336 (Spring 2026) course page](https://cs336.stanford.edu/): In Assignment 2 (Systems), students implement FlashAttention2 in Triton.
  - community · [Sebastian Raschka, LLM reading list](https://www.sebastianraschka.com/blog/2023/llm-reading-list.html): Item 6, under 'Scaling Laws and Improving Efficiency'.
  - survey · [Zhao et al., A Survey of Large Language Models](https://arxiv.org/html/2303.18223): Presents FlashAttention as the attention optimization that, unlike approximate methods, keeps quality while cutting GPU speed and memory costs. Notes that it …
  - adoption · [LLaMA paper, Section 2.4](https://arxiv.org/html/2302.13971): LLaMA's efficient causal attention 'uses the backward from Dao et al. (2022)'.
- **LLaMA** (2023): Combined pre-RMSNorm, SwiGLU and RoPE in a decoder-only model shared with researchers; later open models follow its recipe.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2302.13971?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 22,069 citations, 2,252 influential (not re-fetched).
  - adoption · [DeepSeek LLM paper (arXiv 2401.02954)](https://arxiv.org/html/2401.02954): 'The micro design of DeepSeek LLM largely follows the design of LLaMA' (not re-fetched). DeepSeek-V2 then follows DeepSeek 67B for its normalization and FFN …
  - community · [Sebastian Raschka, The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison): Notes how 'structurally similar' 2025 flagship models still are, built on the same refinements (RoPE, RMSNorm, SwiGLU, GQA).
  - syllabus · [Stanford CS336 (Spring 2026) Lecture 3 (third-party summary)](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-lVynu4/): Describes LLaMA as a model line that shaped many later architectures.
- **GQA** (2023): Shares each key/value head across a group of query heads, keeping quality while shrinking the KV cache.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2305.13245?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 1,920 citations, 145 influential (not re-fetched).
  - syllabus · [Stanford CS336 (Spring 2026) Lecture 3 (third-party summary)](https://summify.io/discover/stanford-cs336-language-modeling-from-scratch-spring-2026-le-lVynu4/): 'Grouped Query Attention (GQA) offers a favorable trade-off between inference cost and performance.'
  - survey · [Zhao et al., A Survey of Large Language Models](https://arxiv.org/html/2303.18223): Presents GQA as the trade-off between multi-query and multi-head attention and notes its adoption in LLaMA 2.
  - adoption · [Sebastian Raschka, The Big LLM Architecture Comparison; DeepSeek-V4 report](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison): 'Llama 4 uses Grouped-Query Attention similar to its predecessors.' DeepSeek-V4 uses BF16 GQA8 as its KV-cache baseline.
- **DeepSeekMoE** (2024): Split experts into many smaller ones and added always-on shared experts so routed experts specialize; the MoE in DeepSeek-V2/V3.
  - arxiv · [arXiv 2401.06066](https://arxiv.org/abs/2401.06066): v1 Thu, 11 Jan 2024 17:31:42 UTC, only version. Title: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.
  - venue · [ACL Anthology](https://aclanthology.org/2024.acl-long.70/): Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers), August 2024, pp. 1280-1297.
  - adoption · [DeepSeek-V2 technical report](https://arxiv.org/abs/2405.04434): DeepSeek-V2 Sec 2.2.1: 'For FFNs, we employ the DeepSeekMoE architecture (Dai et al., 2024).' DeepSeek-V3 Sec 2.1.2 (arXiv 2412.19437): 'For Feed-Forward …
  - survey · [Cai et al., A Survey on Mixture of Experts in LLMs (arXiv 2407.06204)](https://arxiv.org/abs/2407.06204): Cai et al.: 'Differently, DeepSeekMoE [30], [67] introduces the concept of fine-grained expert segmentation ...' and 'Qwen1.5-MoE [102] and DBRX [28] adopt a …
  - syllabus · [Stanford CS336 (Spring 2025) Lecture 4: MoEs](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf): CS336 L4 closes with 'DeepSeek MoE v1-v2-v3 ... To wrap up, we'll walk through the DeepSeek MoE architecture.' Its [Dai et al 2024] slide reads: 'Smaller …
- **DeepSeek-V2** (2024): Introduced Multi-head Latent Attention, compressing the KV cache into a low-rank latent, in a large MoE model.
  - citations · [Semantic Scholar (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2405.04434?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 1,541 citations, 143 influential (not re-fetched).
  - community · [Sebastian Raschka, The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison): 'MLA is not new in DeepSeek V3, as its DeepSeek-V2 predecessor also used (and even introduced) it.'
  - adoption · [Kimi K3 technical report, Section 2.1.2](https://arxiv.org/html/2607.24653v1): 'Multi-head Latent Attention (MLA), introduced in DeepSeek-V2' was 'subsequently adopted by Kimi K2 and Kimi K2.5', and K3 keeps it in its global-attention …
  - adoption · [PyTorch blog, Using Muon Optimizer with DeepSpeed (Jun 3, 2026)](https://pytorch.org/blog/using-muon-optimizer-with-deepspeed/): GLM-5 uses a 'Muon Split' technique that splits MLA up-projection matrices by head, which confirms MLA in GLM-5.
- **DeepSeek-V3** (2024): Trained a 671B-parameter MoE cheaply, adding auxiliary-loss-free load balancing and multi-token prediction.
  - citations · [Semantic Scholar DBLP-linked record (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/search?query=DeepSeek-V3+Technical+Report&fields=title,year,citationCount,influentialCitationCount,externalIds&limit=3): 4,271 citations, 443 influential (not re-fetched). The arXiv-linked duplicate shows 0.
  - community · [Sebastian Raschka, The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison): 'Kimi K2 is basically the same as DeepSeek V3, except that it uses more experts in the MoE modules.'
  - community · [Forklog report embedding Andrej Karpathy's Dec 26, 2024 post](https://forklog.com/en/chinese-startup-deepseek-unveils-ai-model-surpassing-meta-and-openai/): Karpathy: frontier-grade capability 'trained on a joke of a budget (2048 GPUs for 2 months, $6M)'.
  - adoption · [DeepSeek-V4 report; Kimi K3 report](https://arxiv.org/html/2606.19348v1): V4 adopts DeepSeekMoE and the auxiliary-loss-free strategy, and keeps an MTP configuration 'identical to that of DeepSeek-V3'. Kimi K3 'adopts …

**Arrows** (the relation sentence shown on the site, then its source and quote):

- Transformer → GPT-3: GPT-3 scaled up the Transformer and found it could pick up new tasks from a few examples. ([source](https://arxiv.org/abs/2005.14165): “We use the same model and architecture as GPT-2 [RWC+19], including the modified initialization, pre-normalization, and reversible tokenization described …”)
- RMSNorm → LLaMA: LLaMA used RMSNorm to normalize the input of every layer, keeping training stable. ([source](https://arxiv.org/abs/2302.13971): “To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing …”)
- RoPE → LLaMA: LLaMA dropped absolute position embeddings and applied RoPE at every layer instead. ([source](https://arxiv.org/abs/2302.13971): “We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by Su et al. (2021), at each layer of the …”)
- DeepSeek-V2 → DeepSeek-V3: DeepSeek-V3 reused DeepSeek-V2's memory-saving attention and mixture-of-experts design. ([source](https://arxiv.org/abs/2412.19437): “To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were …”)
- GQA → DeepSeek-V2: Like GQA, DeepSeek-V2 shrank a model's memory of earlier text, but without losing quality. ([source](https://arxiv.org/abs/2405.04434): “Grouped-Query Attention (GQA) (Ainslie et al., 2023) and Multi-Query Attention (MQA) (Shazeer, 2019). However, these methods often compromise performance in …”)
- Transformer → FlashAttention: FlashAttention computed Transformer attention in tiles, making it faster yet still exact. ([source](https://arxiv.org/abs/2205.14135): “We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory …”)
- Transformer → SwiGLU: SwiGLU swapped the ReLU in Transformer feed-forward layers for a gate, improving quality. ([source](https://arxiv.org/abs/2002.05202): “We test these variants in the feed-forward sublayers of the Transformer (arXiv:1706.03762) sequence-to-sequence model, and find that some of them yield quality …”)
- Sparse MoE → GShard: GShard moved Sparse MoE's expert layers from LSTM models into the Transformer. ([source](https://arxiv.org/abs/2006.16668): “Scaling capacity of RNN-based machine translation and language models by adding Position-wise Sparsely Gated Mixture-of-Experts (MoE) layers [16] allowed to …”)
- GShard → Switch: Switch cut GShard's top-2 routing to one expert per token, making MoE simpler and cheaper. ([source](https://arxiv.org/abs/2101.03961): “Contrary to these ideas, we instead use a simplified strategy where we route to only a single expert. We show this simplification preserves model quality …”)
- GShard → DeepSeekMoE: DeepSeekMoE cut GShard's experts finer and added shared ones so each expert specializes. ([source](https://arxiv.org/abs/2401.06066): “conventional MoE architectures like GShard, which activate the top-K out of N experts, face challenges in ensuring expert specialization, i.e. each expert …”)
- DeepSeekMoE → DeepSeek-V2: DeepSeek-V2 used DeepSeekMoE for its feed-forward layers to train a strong model cheaply. ([source](https://arxiv.org/abs/2405.04434): “For FFNs, we adopt the DeepSeekMoE architecture (Dai et al., 2024), a high-performance MoE architecture that enables training strong models at an economical …”)
- Transformer → GShard: GShard swapped every other Transformer feed-forward layer for a mixture of experts. ([source](https://arxiv.org/abs/2006.16668): “We sparsely scale Transformer with conditional computation by replacing every other feed-forward layer with a Position-wise Mixture of Experts (MoE) layer [16] …”)

**Research core papers not on the roadmap:** AdamW (high), Chinchilla (high), Muon (Muon is Scalable) (med), DeepSeek-V4 (med).

**Collection members not on the roadmap** (research-stage reasons):

- `adamw`: A research core paper (AdamW) that is not on the roadmap; see the curation notes.
- `lora`: Parameter-efficient fine-tuning. It does not change the pre-training architecture or training efficiency of frontier base models, and it is already a member of the post-training collection.
- `kimi-k3`: Listed as borderline. It is 12 weeks old and mainly scales Kimi Linear's KDA hybrid together with Attention Residuals, LatentMoE, MLA and Muon, each introduced elsewhere. It is the first candidate to promote, with Gated DeltaNet as its parent, if hybrid linear attention becomes the default.
- `glm-5`: Strong frontier open model (592 Semantic Scholar citations, not re-fetched), but its architecture mostly adopts others' components: DeepSeek Sparse Attention from V3.2, MLA, V3-style MTP, and Muon from GLM-4.5. Its new pieces are small (MLA-256, Muon Split, shared MTP layers). Its main contributions are asynchronous agentic RL infrastructure and algorithms, and it is already a member of the RL collection.
- `es-at-scale`: Added to the collection after the roadmap research; not assessed for the roadmap.
- `neural-thickets`: Added to the collection after the roadmap research; not assessed for the roadmap.

**Borderline candidates** (research-stage reasons):

- Kimi K3 (2026, arXiv 2607.24653): It is 12 weeks old (Jul 27, 2026). Its own report says it is 'built on Kimi Delta Attention [Kimi Linear] and Attention Residuals [a separate 2026 paper]', and Raschka calls it 'essentially a scaled-up production version' of Kimi Linear. By the standard used to exclude GLM-5, it mostly integrates components introduced elsewhere: KDA, AttnRes, NVIDIA's LatentMoE, MLA and Muon. Its distinctive result is real: it is the largest open-weight model and the first frontier-scale linear-attention hybrid (Lukyanenko review), with day-0 SGLang/vLLM support. That is not yet enough for a place on the map, which is at the 14-paper cap. It is the first candidate to add, together with Gated DeltaNet, if hybrid linear attention becomes the default.
- Gated DeltaNet (2024, arXiv 2412.06464): ICLR 2025. It is the linear layer in the 3:1 hybrids of Qwen3-Next and Qwen3.5 (Raschka's hybrid-attention gallery). Kimi Linear's KDA 'builds on Gated DeltaNet' with finer gating, and Kimi K3 cites it (ref. 137) for its delta-rule recurrence. It opens the hybrid-linear mechanism line. Its citation count could not be retrieved.
- Scaling Laws for Neural Language Models (Kaplan et al.) (2020, arXiv 2001.08361): It opened LM power-law scaling (9,359 citations, not re-fetched), GPT-3 was sized in its regime, and both the Kimi K3 and DeepSeek-V4 reports still cite it. Chinchilla, a research core paper that is not on the roadmap, replaced its parameter-heavy allocation and is the result LLaMA builds on; neither scaling-law paper is on the map.
- MQA (Multi-Query Attention) (2019, arXiv 1911.02150): Opened KV-head sharing, and GQA explicitly generalizes it. DeepSeek-V4's CSA/HCA use 'shared key-value MQA', which revives it. GQA remains the variant most open dense models adopted. It is the first candidate if the MQA, GQA, MLA chain should be fully explicit (the map is at the 14-paper cap).
- Mamba (2023, arXiv 2312.00752): The most-cited attention alternative (9,678 citations, not re-fetched) and the root of the selective-SSM line. Mamba-2 layers run NVIDIA's Nemotron 3 hybrids, and Gated DeltaNet is framed as 'Improving Mamba2 with Delta Rule'. It would anchor an attention-alternatives branch above Gated DeltaNet.
- Kimi Linear (2025, arXiv 2510.26692): Introduced KDA, a 48B/3B-active KDA + MLA hybrid that beats full MLA under an identical training recipe, with up to 75% less KV cache at 1M tokens. Kimi K3 scales it. It sits between Gated DeltaNet and Kimi K3 in the hybrid line.
- Gated Attention for LLMs (2025, arXiv 2505.06708): NeurIPS 2025 Best Paper (Qwen team). A sigmoid output gate after attention removes attention sinks and stabilizes training. It ships in Qwen3-Next and Qwen3.5, and Kimi K3 cites it (ref. 98) for its gated MLA layers. It is a block-level refinement, too recent to judge against GQA/MLA-level changes.
- Mixtral of Experts (2024, arXiv 2401.04088): The Cai et al. survey marks it as the 2024 trigger of open MoE growth. Architecturally it is a top-2 MoE on Mistral 7B with no new mechanism, and DeepSeek-V2/V3 defined the MoE design later models copied. On the mixture-of-experts path it is the runner-up and is not included: Semantic Scholar shows 2219 citations, and it prompted the HF post. Its paper says 'This formulation is similar to the GShard architecture', so it adds no mechanism and does not feed DeepSeek-V2. Adding it would exceed the 14-paper cap; it would hang off GShard as a leaf, but auxiliary-loss-free balancing would fit the chain better (see below).
- Kimi K2 (MuonClip) (2025, arXiv 2507.20534): The first 1T-parameter model trained with a Muon variant (MuonClip). Kimi K3 says it adopts Muon 'following Kimi K2'. Architecturally it is 'basically the same as DeepSeek V3' (Raschka), so its architecture is covered by the DeepSeek-V3 node; its optimizer traces to Muon, a research core paper that is not on the roadmap.
- DeepSeek-V3.2 (DeepSeek Sparse Attention) (2025, arXiv 2512.02556): Introduced DSA (a lightning indexer plus top-k token selection). DeepSeek-V4's CSA 'performs DeepSeek Sparse Attention' on compressed KV entries, and GLM-5 adopts DSA. It is the direct mechanism ancestor of V4's attention. DeepSeek-V4, a research core paper that is not on the roadmap, is the preferred endpoint of this attention line because CSA/HCA and the 1M-context efficiency are new there.
- YaRN (2023, arXiv 2309.00071): The most widely used RoPE context-extension method before 2025. Kimi K3 explicitly avoids 'applying YaRN' because it uses NoPE, and DeepSeek-V4 does not mention it. It is a RoPE refinement that 2026 frontier models are moving past.
- Attention Residuals (AttnRes) (2026, arXiv 2603.15031): Replaces fixed residual accumulation with softmax attention over earlier layers and ships in Kimi K3. DeepSeek-V4 instead uses mHC (Xie et al., 2026). It is too new to tell which residual-path redesign, if either, becomes standard.

**Rejected notable candidates** (research-stage reasons):

- On Layer Normalization in the Transformer Architecture (Pre-LN, Xiong et al.): Explains why pre-norm trains more stably, but GPT-2/GPT-3 already used pre-norm and LLaMA attributes it to GPT-3.
- ALiBi: The Zhao survey recommends it alongside RoPE, and BLOOM and MPT used it, but RoPE won. No frontier model on this roadmap uses ALiBi.
- FlashAttention-2 / FlashAttention-3: Engineering refinements of FlashAttention (better parallelism, Hopper features). DeepSeek-V2 builds its MLA kernel on an improved FlashAttention-2, which is implementation lineage. The opening paper is the core node.
- Native Sparse Attention (NSA): ACL 2025 best paper on trainable compression-plus-selection sparse attention. DeepSeek's production line goes through DSA (V3.2), and the DeepSeek-V4 report does not cite NSA, so it is not on the cited path from DeepSeek-V3.2 to DeepSeek-V4.
- QK-Norm: A widely adopted stability trick: CS336 Lecture 3 covers it, and DeepSeek-V4 normalizes queries and KV entries. It has no single opening paper that frontier reports attribute it to.
- Differential Transformer (catalog paper differential-transformer): ICLR 2025 attention variant. It is not cited by the DeepSeek-V4 or Kimi K3 reports, and no frontier open model in scope adopts it. It is neither a member of this collection nor core.
- PagedAttention / vLLM: Very influential for inference serving, but it is a serving-system contribution outside model architecture and training efficiency.
- Megatron-LM and ZeRO: Foundational distributed-training systems (tensor parallelism, optimizer-state sharding). Kimi K3 cites ZeRO, but these would anchor a systems roadmap.
- PaLM: Popularized SwiGLU (LLaMA credits it with '[PaLM]'), MQA and parallel blocks at 540B, but it is a closed model. LLaMA is the open reference that later models explicitly follow.
- Llama 2 / Llama 3 / Llama 4: Incremental over LLaMA: more data, GQA adoption, MoE in Llama 4. None introduced a new mechanism that others copied.
- BERT (catalog paper bert): Encoder-only, while the frontier LLM stack is decoder-only. It belongs on a representation-learning roadmap.
- Mistral 7B / sliding-window attention: Sliding-window attention recurs (Mistral, Gemma 3, and as an auxiliary branch in DeepSeek-V4's CSA/HCA), but it has no single opening paper and is not the main long-context mechanism of 2026 frontier models.
- Qwen3 / GLM-4.5 technical reports: Strong models whose architectures combine existing components (GQA or MLA, QK-norm, DeepSeek-style MoE, Muon) without a new mechanism that others adopted.
- DeepSeek-R1: Field-changing, but its contribution is RL-based reasoning post-training. It belongs on the RL/post-training roadmap.
- GLaM: not included. A decoder-only scale demo with GShard-style top-2 routing and no new mechanism. DeepSeekMoE lists it only among typical practice.
- ST-MoE: not included. The router z-loss is a stability add-on, and no z-loss appears in the DeepSeek-V2/V3 texts. Semantic Scholar shows 486 citations, below the Switch/GShard tier.
- Expert Choice routing: not included. Its Limitations section says: 'The expert choice method might not immediately apply to auto-regressive text generation'. CS336: 'Almost all the MoEs do a standard token choice topk routing'.
- OLMoE: not included. An open recipe and ablation paper (360 Semantic Scholar citations). It postdates DeepSeek-V2 and introduces no mechanism.
- Auxiliary-loss-free load balancing: not included (the strongest further MoE candidate). Already named in the DeepSeek-V3 node sentence. CS336 says '(but the approach is not fully aux loss free..)'. It is the only candidate that would add a real arrow into DeepSeek-V3, which has room for one more incoming arrow (DeepSeek-V3: 'pioneers an auxiliary-loss-free strategy (Wang et al., 2024a)'). arXiv v1 is 2024-08-28.
- DeepSpeed-MoE: not included. DeepSeekMoE explicitly credits it as the 'prototype of shared expert isolation', and CS336 says 'originally from DeepSpeed MoE'. It is left out nonetheless: it is mainly a systems/inference paper (PR-MoE, MoS, inference system). DeepSeek-V2/V3 adopt the DeepSeekMoE design, and fine-grained segmentation, the other half of that design, is not in DeepSpeed-MoE. Semantic Scholar shows 684 citations.
- MegaBlocks: not included. A dropless MoE kernel library. CS336 says it is 'used in many open MoEs', but it is a systems contribution, not on the architectural chain into DeepSeek.
- Mesh-TensorFlow: not included. Switch App. A says Shazeer et al. (2018) designed MoE Transformers before GShard. It is a library paper whose abstract does not mention MoE. The survey and the DeepSeek papers credit GShard, so it does not change GShard's place on the path.

**Curation notes:** The map is at the 14-paper cap, so Chinchilla, AdamW, Muon and DeepSeek-V4 are left out and mixture-of-experts gets its own path. The mixture-of-experts papers meet the same evidence rule as the rest of the map. With GShard, Switch Transformer and DeepSeekMoE (all rated high), the map has 14 papers. The path is Sparse MoE → GShard → DeepSeekMoE → DeepSeek-V2 → DeepSeek-V3, with Switch Transformer as a one-step branch off GShard. Switch → DeepSeekMoE is not drawn: DeepSeekMoE builds on GShard and moves to more, finer experts, the opposite of Switch's top-1 routing. Mixtral is the runner-up: its routing follows GShard and it does not feed DeepSeek. Transformer → GShard is drawn because GShard replaces every other Transformer feed-forward layer with an MoE layer. GPT-3 → LLaMA and LLaMA → DeepSeek-V2 are not drawn because LLaMA (RMSNorm, RoPE) and DeepSeek-V2 (GQA, DeepSeekMoE) already have two incoming arrows.

## Retrieval & Recommendation

Two communities on one map. Recommendation runs from BPR through sequential (GRU4Rec, SASRec), graph (LightGCN) and generative (TIGER) recommenders. Text retrieval covers BERT re-ranking, dense retrieval and late interaction (monoBERT, DPR, ColBERT).

| Paper | Year | Venue | Exercise | Confidence |
| --- | --- | --- | --- | --- |
| [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) | 2009 | UAI 2009 | `bpr` | high |
| [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939) | 2015 | ICLR 2016 | grey | high |
| [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) | 2018 | ICDM 2018 | `sasrec` | high |
| [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) | 2019 | arXiv | grey | high |
| [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) | 2020 | SIGIR 2020 | `lightgcn` | high |
| [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | 2020 | EMNLP 2020 | grey | high |
| [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) | 2020 | SIGIR 2020 | `colbert` | high |
| [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) | 2023 | NeurIPS 2023 | `tiger` | high |

**Evidence per paper** (type, linked source, shortened note):

- **BPR** (2009): Learned to rank items a user interacted with above ones they did not; its pairwise loss is still a default.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:1205.2618?fields=title,year,venue,citationCount,influentialCitationCount): 6,990 citations and 1,491 influential citations (re-fetched), a very high influential share.
  - community · [RecSys 2026 '20x20: Top 20 Milestones' reference guide](https://recommender-systems.com/wp-content/uploads/2026/08/recsys2026_milestones_reference.pdf): BPR is one of seven 'Fundamentals' milestones: 'It became one of the most widely adopted training objectives in recommendation research.'
  - adoption · [LightGCN (SIGIR 2020)](https://arxiv.org/pdf/2002.02126): 'We employ the Bayesian Personalized Ranking (BPR) loss [32]'. NGCF (arXiv 1905.08108) says all compared methods optimize the BPR loss.
  - adoption · [GRU4Rec (ICLR 2016)](https://arxiv.org/pdf/1511.06939): Uses BPR, applied over several sampled negatives, as one of its two pairwise ranking losses.
- **GRU4Rec** (2015): The first to model a user's session with a recurrent network to predict the next item.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:1511.06939?fields=title,year,venue,citationCount,influentialCitationCount): 3,216 citations and 558 influential citations (re-fetched).
  - community · [RecSys 2026 '20x20: Top 20 Milestones' reference guide](https://recommender-systems.com/wp-content/uploads/2026/08/recsys2026_milestones_reference.pdf): Sequential-models milestone: GRU4Rec 'was the first work to successfully apply recurrent neural networks to session-based recommendation ... This opened the …
  - survey · [Pan et al., A Survey on Sequential Recommendation (2025)](https://arxiv.org/pdf/2412.12770): Presents GRU4Rec as the representative RNN-based sequential recommender, the base of GRU4Rec+ and later RNN models.
  - other · [HSTU paper (ICML 2024), Appendix B](https://arxiv.org/pdf/2402.17152): 'Recurrent neural networks (RNNs) were first applied to recommendation scenarios in GRU4Rec'.
- **SASRec** (2018): Replaced RNNs with causal self-attention over the interaction history; the most used sequential baseline.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:1808.09781?fields=title,year,venue,citationCount,influentialCitationCount): 4,375 citations and 1,141 influential citations (re-fetched).
  - survey · [Pan et al., A Survey on Sequential Recommendation (2025)](https://arxiv.org/pdf/2412.12770): 'SASRec [1] is currently the most popular SR model'.
  - community · [RecSys 2026 '20x20: Top 20 Milestones' reference guide](https://recommender-systems.com/wp-content/uploads/2026/08/recsys2026_milestones_reference.pdf): SASRec is named in the sequential-models milestone with GRU4Rec.
  - adoption · [HSTU paper (ICML 2024)](https://arxiv.org/pdf/2402.17152): Uses SASRec as the baseline ('identical configurations as SASRec'), and its appendix concludes 'SASRec remains one of the most competitive approaches in …
  - syllabus · [University of Wroclaw RecSys seminar list (2026)](https://kdy.cs.uni.wroc.pl/recsys): SASRec and LightGCN are the two model papers presented. This is a small student seminar, so it is weak evidence.
- **monoBERT** (2019): Fine-tuned BERT to judge query-passage relevance for re-ranking; the reference cross-encoder.
  - survey · [Lin, Nogueira & Yates, Pretrained Transformers for Text Ranking](https://arxiv.org/pdf/2010.06467): 'The earliest use of BERT for text ranking was reported in January 2019 [Nogueira and Cho, 2019]'. The book has a dedicated monoBERT section.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:1901.04085?fields=title,year,venue,citationCount,influentialCitationCount): 1,691 citations and 186 influential citations (re-fetched) for an arXiv-only paper.
  - syllabus · [Stanford CS224U (2021) IR lecture slides](https://cs224u.stanford.edu/2021/slides/CS224U-IR-part-5.pdf): Cites Nogueira & Cho 2019 as the BERT re-ranker, alongside DPR and ColBERT.
- **LightGCN** (2020): Showed graph recommenders need only linear propagation of ID embeddings over the user-item graph.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2002.02126?fields=title,year,venue,citationCount,influentialCitationCount): 5,852 citations and 958 influential citations (re-fetched).
  - syllabus · [Stanford CS224W, lecture 11 'GNNs for recommender systems'](https://web.stanford.edu/class/cs224w/): The lecture readings are NGCF, LightGCN and PinSage (confirmed on the current course page).
  - syllabus · [University of Wroclaw RecSys seminar list (2026)](https://kdy.cs.uni.wroc.pl/recsys): LightGCN is presented alongside SASRec (weak evidence).
- **DPR** (2020): A BERT dual encoder trained on question-passage pairs that beat BM25, making dense retrieval practical.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2004.04906?fields=title,year,venue,citationCount,influentialCitationCount): 7,366 citations and 1,098 influential citations (re-fetched).
  - survey · [Lin, Nogueira & Yates, Pretrained Transformers for Text Ranking](https://arxiv.org/pdf/2010.06467): The dense-retrieval chapter is built around bi-encoders such as DPR and ANCE.
  - syllabus · [Stanford CS224U (2021) IR lecture slides](https://cs224u.stanford.edu/2021/slides/CS224U-IR-part-5.pdf): DPR is the worked example of representation-similarity retrieval.
  - adoption · [Lewis et al., Retrieval-Augmented Generation (NeurIPS 2020)](https://arxiv.org/pdf/2005.11401): 'The retrieval component p_eta(z|x) is based on DPR'.
- **ColBERT** (2020): Late interaction: encode query and document tokens separately, then score by summing the best token matches.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2004.12832?fields=title,year,venue,citationCount,influentialCitationCount): 2,664 citations and 284 influential citations (re-fetched).
  - survey · [Lin, Nogueira & Yates, Pretrained Transformers for Text Ranking](https://arxiv.org/pdf/2010.06467): Has a dedicated section on per-token representations and late interaction (ColBERT).
  - community · [LIR: First Workshop on Late Interaction and Multi-Vector Retrieval @ ECIR 2026](https://arxiv.org/abs/2511.00444): 'Late interaction retrieval methods, pioneered by ColBERT, have emerged as a powerful alternative to single-vector neural IR.'
  - adoption · [vLLM documentation, ColQwen3 model](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/colqwen3/): vLLM serves ColBERT-style late-interaction models (ColQwen3, per-token embeddings scored with MaxSim). Seen in search results.
- **TIGER** (2023): Represents items as semantic ID codes and generates the next item's code: generative retrieval for recommendation.
  - survey · [Pan et al., A Survey on Sequential Recommendation (2025)](https://arxiv.org/pdf/2412.12770): '[TIGER] is the first model that leverages semantic IDs for generative recommendation'.
  - survey · [Ju et al. (Snap), Generative Recommendation with Semantic IDs: A Practitioner's Handbook](https://arxiv.org/html/2507.22224v1): 'TIGER (Rajput et al., 2023) first applied transformers to predict item SIDs for recommendation, extending ideas from GR in document retrieval (Tay et al. …
  - adoption · [NVIDIA Technical Blog, How Generative Recommenders Are Redefining RecSys at Scale (2026-08-20)](https://developer.nvidia.com/blog/how-generative-recommenders-are-redefining-recsys-at-scale/): 'Architectures such as TIGER, PLUM, OneRec v1/v2' use Semantic IDs as their foundation.
  - citations · [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/arXiv:2305.05065?fields=title,year,venue,citationCount,influentialCitationCount): 483 citations and 98 influential citations (re-fetched), high for a 2023 recsys paper.

**Arrows** (the relation sentence shown on the site, then its source and quote):

- BPR → LightGCN: LightGCN trained its embeddings with BPR's loss, ranking a user's items above the rest. ([source](https://arxiv.org/abs/2002.02126): “We employ the Bayesian Personalized Ranking (BPR) loss [32], which is a pairwise loss that encourages the prediction of an observed entry to be higher than its …”)
- BPR → GRU4Rec: GRU4Rec borrowed BPR's pairwise ranking loss to train its RNN on user sessions. ([source](https://arxiv.org/abs/1511.06939): “BPR: Bayesian Personalized Ranking (Rendle et al., 2009) is a matrix factorization method that uses pairwise ranking loss. ... We found the ones introduced in …”)
- GRU4Rec → SASRec: SASRec replaced GRU4Rec's step-by-step RNN with self-attention over all past items. ([source](https://arxiv.org/abs/1808.09781): “the computation in each self-attention layer is fully parallelizable, which is amenable to GPU acceleration. In contrast, RNN-based methods (e.g. GRU4Rec [2]) …”)
- SASRec → TIGER: TIGER kept SASRec's next-item task but generated semantic IDs instead of vector search. ([source](https://arxiv.org/abs/2305.05065): “SASRec [17] ... perform an ANN in a Maximum Inner Product Search (MIPS) space to predict the next item. In contrast, our proposed technique, TIGER, uses …”)
- monoBERT → ColBERT: ColBERT encoded query and passage separately for speed; monoBERT ran BERT on every pair. ([source](https://arxiv.org/abs/2004.12832): “(and the one adopted by Nogueira et al. on MS MARCO and TREC CAR) is to feed the query–document pair through BERT ... independently encodes the query and the …”)

**Research core papers not on the roadmap:** Matrix Factorization (Netflix Prize) (high), FPMC (high), YouTube DNN (high), HSTU (Generative Recommenders) (med), DSSM (high), DSI (med), CLIP (high).

**Collection members not on the roadmap** (research-stage reasons):

- `graphsage`: A general inductive GNN evaluated on citation, Reddit and PPI graphs. It is enormously cited (S2: 21,620 citations, re-fetched), but not as a recommender. Its recommendation impact arrives through PinSage, which NGCF describes as 'designed to employ GraphSAGE'. The graph-CF line that became standard (NGCF -> LightGCN) does not build on it directly. It belongs on a graph/representation roadmap, with a cross-link from the borderline PinSage.
- `clip`: A research core paper (CLIP) that is not on the roadmap; see the curation notes.
- `rotation-trick`: A gradient estimator for vector-quantized autoencoders (68 S2 citations, not re-fetched). In retrieval it appears only as an RQ-VAE tokenizer baseline in semantic-ID papers: QuaSID (arXiv 2603.00632) lists 'RQ-VAE-Rotation [6]', which 'replaces the STE in RQ-VAE with the rotation trick'. There is no evidence that it became standard in semantic-ID pipelines. It belongs on a representation or generative-models roadmap.

**Borderline candidates** (research-stage reasons):

- NGCF (2019, arXiv 1905.08108): Direct parent of LightGCN, which ablates it, and a CS224W reading (S2: 4,127 citations, 653 influential). LightGCN showed that NGCF's GCN-inherited components are unnecessary and replaced it as the standard model. It would fit only if the roadmap were to show the simplification story.
- Factorization meets the neighborhood (Koren 2008) (2008): The RecSys 2026 '20x20 milestones' guide names it the fundamentals paper of the latent-factor era (OpenAlex: 3,854 citations), and FPMC and GRU4Rec cite it. It is an alternative root for the recommendation line. The research preferred Koren et al. 2009 (Netflix MF) as the more cited (11,271) and more readable summary of the same model family; neither has an arXiv version, so neither is on the roadmap.
- Implicit-feedback ALS (Hu, Koren, Volinsky) (2008): Confidence-weighted MF for implicit data (S2: 3,628 citations, 516 influential) and the basis of Spark ALS implicitPrefs. An IEEE Computer Society announcement (seen in search results) says it won the 2017 ICDM 10-Year Highest-Impact Paper Award, and FPMC cites it as one of the MF models it builds on. Its role overlaps with BPR and with Netflix MF, which is not on the roadmap because it has no arXiv version.
- PinSage (2018, arXiv 1806.01973): First web-scale GCN recommender, deployed at Pinterest (S2: 4,335 citations, 231 influential) and a CS224W reading. It is an industrial milestone, but the academic graph-CF line (NGCF -> LightGCN) only uses it as a baseline. It is the candidate that would link GraphSAGE, a collection member that is not on the roadmap, if the roadmap added a graph-industry paper.
- NCF (Neural Collaborative Filtering) (2017, arXiv 1708.05031): Very highly cited (S2: 7,440 citations, 1,017 influential), and it triggered the academic wave of neural CF. However, Rendle et al. (RecSys 2020, arXiv 2005.09683) showed that a tuned dot product beats its learned MLP similarity, and it is absent from the RecSys 2026 milestone list, which picks YouTube DNN. Strongest candidate if the roadmap wants an academic neural-CF node.
- Wide & Deep (2016, arXiv 1606.07792): Deployed in Google Play and very highly cited (4,341 S2 citations, not re-fetched). It opened deep CTR ranking, which is outside the stated scope; it should be core if a ranking/CTR line is added.
- BERT4Rec (2019, arXiv 1904.06690): S2: 3,684 citations, 429 influential. It is a bidirectional, masked-item refinement of SASRec, and its reported gains were not consistently reproduced (Petrov & Macdonald RecSys 2022; Klenitskiy & Vasilev RecSys 2023). HSTU's appendix still finds SASRec 'one of the most competitive approaches'.
- Sentence-BERT (2019, arXiv 1908.10084): S2: 21,429 citations and 2,544 influential, about three times DPR. Its sentence-transformers library is the most common way to run embedding search, and CS224U lists SBERT with DPR as BERT-based representation-similarity models. It stays borderline because it targets sentence similarity and DPR does not build on it, but it is the strongest dense-retrieval candidate if the text-retrieval line widens to general embedding models.
- RAG (2020, arXiv 2005.11401): Huge adoption and directly built on DPR ('The retrieval component ... is based on DPR'), but its contribution is retrieval-conditioned generation. It belongs on an LLM/NLP roadmap, with a cross-link from DPR.
- ColPali (2024, arXiv 2407.01449): ICLR 2025 (S2: 251 citations, 65 influential). It extends ColBERT's late interaction to page images with a vision-language model, and ColQwen models are now served by vLLM. It is the natural bridge from ColBERT to multimodal late-interaction retrieval, but too recent to call a line-opener; worth revisiting in 2027.
- Semantic IDs for ranking (YouTube) (2023, arXiv 2306.08121): Concurrent with TIGER; it uses RQ-VAE Semantic IDs to replace random item IDs in a production YouTube ranker. It is strong adoption evidence for semantic IDs, but surveys and the Snap handbook cite TIGER as the origin of SID generative recommendation.
- OneRec (2025, arXiv 2502.18965): An end-to-end generative recommender from Kuaishou, building on the semantic-ID line; the only evidence cited is the later OneRec Technical Report (arXiv 2506.13695), which reports serving 25% of total QPS. That report had only 32 S2 citations. It is too recent to judge whether it opens a new line or refines TIGER/HSTU; worth revisiting in 2027.
- FAISS (billion-scale ANN) (2017, arXiv 1702.08734): The ANN infrastructure used by ColBERT, DPR and Facebook's embedding-based retrieval. It is an indexing system rather than a retrieval model, better cited in node text than shown as a node.
- Sampling-bias-corrected two-tower (Yi et al.) (2019): The standard logQ-corrected in-batch softmax for industrial two-tower retrieval. It is an important refinement of the YouTube DNN candidate generator, not a new line.

**Rejected notable candidates** (research-stage reasons):

- GroupLens / item-based CF (Sarwar 2001) / Amazon item-to-item (2003): Neighborhood-CF prehistory before the 2009 latent-factor models. Item-based CF won the 2016 Seoul Test of Time Award, and the RecSys 2026 milestone guide lists GroupLens and Amazon item-to-item as application milestones. They have no direct inspiration edges into the roadmap papers.
- Shani et al. 2005 (MDP-based recommender): The RecSys 2026 milestone guide lists it with GRU4Rec and SASRec under sequential models. It predates the 2009 latent-factor models, and FPMC is the Markov-chain model that SASRec explicitly generalizes.
- Factorization Machines (Rendle 2010): Highly cited (S2: 3,148) and generalizes MF to arbitrary feature vectors, but it is the root of the CTR/feature-interaction line (FFM, DeepFM), which is outside the stated scope. It should be core if a CTR line is added.
- BM25: The classical lexical baseline that every neural IR node compares against; prehistory for the text-retrieval line.
- GC-MC: A single graph-convolution layer on the rating graph, superseded by NGCF and LightGCN, which explicitly contrast with it.
- Caser: A CNN sequential recommender from the same period as SASRec, and outperformed by it; a refinement-level contribution.
- ANCE: An asynchronous hard-negative training refinement of DPR-style bi-encoders. Lin et al. present it alongside DPR as a training improvement, not a new paradigm.
- Contriever / E5 / BGE embedding models: Training-recipe and scale refinements of dense retrieval. They are widely used, but no single one opened a line.
- ColBERTv2 / PLAID: Compression and serving refinements of ColBERT's late interaction, best noted in the ColBERT node text.
- SPLADE: Learned sparse retrieval is a separate line not listed in this collection's scope. It would be reconsidered if a sparse-retrieval line is added.
- ALIGN: Concurrent with CLIP and with the same dual-encoder recipe, but far less adopted in open tooling than CLIP, a research core paper that is not on the roadmap.
- DIN / DeepFM / DCN: CTR-ranking refinements outside the stated scope; they would sit under Wide & Deep in a future ranking line.
- P5 and LLM-as-recommender work: No evidence yet of it becoming a default in deployed systems; the semantic-ID and HSTU lines have far stronger adoption evidence.
- GENRE (autoregressive entity retrieval): A generative entity-retrieval precursor that TIGER and DSI cite. TIGER and the GR survey credit DSI with structured semantic docids and call it the seminal GR work.
- Dacrema et al. 2019 / Rendle et al. 2020 (NCF vs MF revisited): Important evaluation critiques that shaped recsys methodology, but they are not methods. Rendle et al. 2020 is cited in the NCF borderline note.
- HNSW: ANN graph-index infrastructure, like FAISS; not a retrieval or recommendation model.

**Curation notes:** Selection rule: high-confidence papers first, then papers that start or connect a line. The research kept 15 papers, 13 of them high. HSTU and DSI (medium) are left out. CLIP is left out as the only high paper with no arrow in the research; it remains on the collection list. Netflix MF, FPMC, DSSM and YouTube DNN are left out because every roadmap paper must have an arXiv version. FPMC → SASRec, Netflix MF → LightGCN and DSSM → ColBERT are therefore not drawn. DPR is shown without arrows: the research has no arrow between DPR and the other roadmap papers.

## Reinforcement Learning

Policy-gradient reinforcement learning and its path to RL for language models: the trust-region and advantage-estimation line that led to PPO, RLHF (Christiano, Ziegler, InstructGPT), and reasoning-oriented RL with GRPO and its successors.

| Paper | Year | Venue | Exercise | Confidence |
| --- | --- | --- | --- | --- |
| [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) | 2015 | ICML 2015 | grey | high |
| [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | 2015 | ICLR 2016 | `gae` | high |
| [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | NeurIPS 2017 | grey | high |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 2017 | arXiv | `ppo` | high |
| [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) | 2019 | arXiv | grey | med |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | 2022 | NeurIPS 2022 | grey | high |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | 2024 | arXiv | `deepseekmath` | high |
| [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | 2025 | Nature 2025 | grey | high |
| [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) | 2025 | NeurIPS 2025 | `dapo` | high |
| [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) | 2025 | COLM 2025 | `dr-grpo` | med |

**Evidence per paper** (type, linked source, shortened note):

- **TRPO** (2015): Updates neural policies by maximizing an importance-ratio objective inside a KL trust region.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1502.05477?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 8,434 citations, 1,185 influential.
  - community · [OpenAI Spinning Up, Key Papers in Deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html): Listed under Policy Gradients together with A3C, GAE and PPO.
  - syllabus · [Berkeley CS285 Lecture 9, Advanced Policy Gradients (Fall 2020 slides)](https://rail.eecs.berkeley.edu/deeprlcourse-fa20/static/slides/lec-9.pdf): The distribution-change bound is 'Proof based on: Schulman ... Trust Region Policy Optimization'; the lecture ends with natural policy gradient, TRPO and PPO …
  - survey · [A Survey of RL for Large Reasoning Models](https://arxiv.org/html/2509.08827v3): 'PPO is proposed as a clipped surrogate objective of TRPO, which preserves the conservative policy iteration of TRPO'.
- **GAE** (2015): A lambda-weighted advantage estimator that trades bias for variance; the default inside PPO.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1506.02438?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 5,064 citations, 634 influential; venue ICLR.
  - adoption · [DeepSeekMath paper, section on PPO and GRPO](https://arxiv.org/pdf/2402.03300): Describes PPO for LLMs as computing advantages 'by applying Generalized Advantage Estimation (GAE) (Schulman et al., 2015)' from a learned value function.
  - survey · [RLHF book, Policy Gradients chapter](https://rlhfbook.com/c/11-policy-gradients): Calls GAE 'the state-of-the-art and canonical implementation in modern systems' for advantage estimation.
  - community · [OpenAI Spinning Up key papers; Cameron Wolfe, GRPO overview](https://cameronrwolfe.substack.com/p/grpo): Wolfe: 'The critic can be used to compute the advantage via Generalized Advantage Estimation (GAE)'. Spinning Up lists GAE among its key policy-gradient papers.
- **Christiano RLHF** (2017): Learned a reward model from human comparisons of two behaviors and used it instead of a hand-written reward for RL.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03741?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 6,200 citations, 688 influential; venue NeurIPS (NIPS 30, 2017).
  - syllabus · [Stanford CS224R schedule](https://cs224r.stanford.edu/): Assigned reading for the 'Reward Learning' lecture.
  - survey · [RLHF book, Key Related Works](https://rlhfbook.com/c/02-related-works): Presented as the application of RLHF to preferences between agent trajectories in Atari, the root of the language-model work.
  - other · [InstructGPT paper, introduction](https://arxiv.org/pdf/2203.02155): Defines its method as 'reinforcement learning from human feedback (RLHF; Christiano et al., 2017; Stiennon et al., 2020)'.
- **PPO** (2017): Replaced TRPO's constraint with a clipped objective; the default optimizer for RLHF and the template GRPO modifies.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1707.06347?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 32,340 citations, 5,333 influential, the highest in this field.
  - adoption · [Cameron Wolfe, GRPO overview](https://cameronrwolfe.substack.com/p/grpo): 'Most early work on RL for LLMs used Proximal Policy Optimization (PPO) as the default RL optimizer.'
  - syllabus · [Stanford CS224R schedule](https://cs224r.stanford.edu/): Assigned reading for the Actor-Critic lecture.
  - survey · [RLHF book, Policy Gradients chapter](https://rlhfbook.com/c/11-policy-gradients): PPO is the central algorithm. TRPO is described as 'accepted as the predecessor to PPO', and GRPO, GSPO and CISPO are presented as modifications of PPO's …
- **Ziegler RLHF** (2019): First applied a preference reward model plus PPO to a pretrained language model, with a KL penalty to the initial model.
  - survey · [RLHF book, Key Related Works](https://rlhfbook.com/c/02-related-works): Calls it 'the earliest work' in the language-model RLHF line, with 'many striking similarities to modern work on RLHF', and says canonical terms were …
  - other · [InstructGPT paper](https://arxiv.org/pdf/2203.02155): 'Our methodology follows that of Ziegler et al. (2019) and Stiennon et al. (2020)'.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1909.08593?fields=title,year,venue,citationCount,influentialCitationCount): 2,790 citations, 213 influential. Stiennon 2020 has more (3,575 / 350); see borderline.
  - community · [Sebastian Raschka, The State of RL for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training): Lists it among the papers that predate InstructGPT: it 'illustrates the concept of PPO and reward learning to pretrained language models including KL …
- **InstructGPT** (2022): Applied SFT, a reward model and PPO to GPT-3 for following instructions; the standard RLHF pipeline.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2203.02155?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 25,112 citations, 2,509 influential; venue NeurIPS 2022.
  - adoption · [RLHF book, Key Related Works (quoting the ChatGPT announcement)](https://rlhfbook.com/c/02-related-works): Quotes OpenAI saying ChatGPT used 'the same methods as InstructGPT' with 'slight differences in the data collection setup'.
  - community · [Sebastian Raschka, The State of RL for LLM Reasoning](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training): Describes InstructGPT as the paper 'which described the recipe that was used to develop the first ChatGPT model'.
  - other · [DeepSeek-R1 and DeepSeekMath papers](https://arxiv.org/pdf/2501.12948): Both describe PPO as 'widely used in the RL stage of LLMs (Ouyang et al., 2022)' before introducing GRPO.
- **GRPO** (2024): Dropped PPO's value network: each answer's advantage is its reward normalized within a group of answers.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2402.03300?fields=title,year,venue,citationCount,influentialCitationCount): 10,151 citations, 2,974 influential. The influential share is very high for a 2024 paper.
  - adoption · [TRL paper index and trainer list; verl README](https://huggingface.co/docs/trl/main/paper_index): GRPOTrainer is one of TRL's main trainers (GRPO and RLOO are the online RL trainers in its trainer list), and its loss options cover DAPO, Dr. GRPO, GSPO and …
  - survey · [A Survey of RL for Large Reasoning Models](https://arxiv.org/html/2509.08827v3): Uses GRPO as the base that 'DAPO, CISPO, Dr. GRPO, LitePPO' modify 'with careful tuning of sampling strategy, clipping threshold, and loss normalization'.
  - community · [Interconnects (Nathan Lambert), Recent reasoning research](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo): Points to DeepSeekMath as the source of GRPO but cautions that 'GRPO is not a special RL algorithm': it derives from PPO and has an advantage computation …
- **DeepSeek-R1** (2025): Showed large-scale GRPO with verifiable rewards on a base model yields long reasoning without SFT.
  - survey · [A Survey of RL for Large Reasoning Models](https://arxiv.org/html/2509.08827v3): Names OpenAI o1 and DeepSeek-R1 as the 'two recent milestones' showing that RLVR can produce long-form reasoning, and calls R1 'the first open-source model to …
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2501.12948?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 5,902 citations, 951 influential in about 21 months. Semantic Scholar indexes it under the Nature title.
  - community · [Interconnects (Nathan Lambert)](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo): 'DeepSeek's results kick-started this revolution.'
  - other · [arXiv record (journal-ref)](https://arxiv.org/abs/2501.12948): Nature 645, 633-638 (2025), DOI 10.1038/s41586-025-09422-z. v2 revised 4 Jan 2026.
- **DAPO** (2025): An open, reproducible large-scale GRPO recipe with clip-higher, dynamic sampling and a token-level loss.
  - adoption · [TRL GRPOTrainer docs](https://huggingface.co/docs/trl/main/grpo_trainer): GRPOConfig defaults to loss_type='dapo' (token-level normalization), and epsilon_high=0.28 is recommended per DAPO.
  - adoption · [verl README](https://github.com/volcengine/verl): Lists DAPO among supported algorithms.
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2503.14476?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 2,981 citations, 609 influential; venue NeurIPS 2025 (also confirmed in the NeurIPS 2025 proceedings).
  - community · [Cameron Wolfe, 'GRPO++: Tricks for Making RL Actually Work'; Interconnects](https://cameronrwolfe.substack.com/p/grpo-tricks): Wolfe opens his survey of GRPO improvements with DAPO's 'four key changes to the vanilla GRPO algorithm'. Lambert calls DAPO 'pretty much GRPO++'.
- **Dr. GRPO** (2025): Showed GRPO's length and std normalizations bias training, and removed both for an unbiased objective.
  - adoption · [TRL GRPOTrainer docs and verl README](https://huggingface.co/docs/trl/main/grpo_trainer): TRL offers loss_type='dr_grpo' and cites the paper for scale_rewards=False (question-level difficulty bias), although the default remains …
  - citations · [Semantic Scholar API (fetched 2026-10-20)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2503.20783?fields=title,year,venue,citationCount,influentialCitationCount,publicationDate): 1,486 citations, 205 influential.
  - other · [COLM 2025 accepted papers](https://colmweb.org/2025/AcceptedPapers.html): Listed as a Spotlight paper; the v2 PDF header reads 'Published as a conference paper at COLM 2025'.
  - survey · [RLHF book, Policy Gradients chapter; RL-for-LRMs survey](https://rlhfbook.com/c/11-policy-gradients): The RLHF book shows that Dr. GRPO's advantage equals RLOO's up to a constant factor. The LRM survey discusses it under 'The Struggle Between De-biasing and …

**Arrows** (the relation sentence shown on the site, then its source and quote):

- TRPO → PPO: PPO kept TRPO's small, safe policy steps but used a simple clipped objective instead. ([source](https://arxiv.org/abs/1707.06347): “introducing an algorithm that attains the data efficiency and reliable performance of TRPO, while using only first-order optimization. We propose a novel …”)
- GAE → PPO: PPO computed each action's advantage with a truncated version of GAE. ([source](https://arxiv.org/abs/1707.06347): “Generalizing this choice, we can use a truncated version of generalized advantage estimation, which reduces to Equation (10) when λ = 1”)
- Christiano RLHF → Ziegler RLHF: Ziegler et al. took Christiano et al.'s preference reward model from simulators to text. ([source](https://arxiv.org/abs/1909.08593): “this work has only recently been applied to modern deep learning, and even then has only been applied to relatively simple simulated environments (Christiano …”)
- Ziegler RLHF → InstructGPT: InstructGPT reused Ziegler's reward model plus PPO recipe, now for general instructions. ([source](https://arxiv.org/abs/2203.02155): “Our methodology follows that of Ziegler et al. (2019) and Stiennon et al. (2020), who applied it in the stylistic continuation and summarization domains.”)
- PPO → InstructGPT: InstructGPT used PPO to fine-tune GPT-3 to maximize its learned reward model's score. ([source](https://arxiv.org/abs/2203.02155): “Finally, we use this RM as a reward function and fine-tune our supervised learning baseline to maximize this reward using the PPO algorithm (Schulman et al. …”)
- PPO → GRPO: GRPO kept PPO's update but dropped the value network, scoring answers against their group. ([source](https://arxiv.org/abs/2402.03300): “Group Relative Policy Optimization (GRPO), a variant reinforcement learning (RL) algorithm of Proximal Policy Optimization (PPO) (Schulman et al., 2017). GRPO …”)
- GRPO → DeepSeek-R1: DeepSeek-R1 used GRPO to cut the cost of the large-scale RL that trained its reasoning. ([source](https://arxiv.org/html/2501.12948): “To facilitate large-scale RL efficiency, we adopt Group Relative Policy Optimization (GRPO) (Shao et al., 2024). ... It was originally proposed to simplify the …”)
- DeepSeek-R1 → DAPO: DAPO openly reproduced and improved how DeepSeek-R1 used rewards to teach reasoning. ([source](https://arxiv.org/abs/2503.14476): “we release an open-source state-of-the-art system for large-scale LLM RL, which achieves 50 points on AIME 2024 based on Qwen2.5-32B model, outperforming …”)
- DeepSeek-R1 → Dr. GRPO: Dr. GRPO studied R1-Zero-style RL and fixed a GRPO bias that made wrong answers longer. ([source](https://arxiv.org/abs/2503.20783): “we critically examine R1-Zero-like training by analyzing its two core components: base models and RL. ... we identify an optimization bias in Group Relative …”)

**Research core papers not on the roadmap:** REINFORCE (high), Policy Gradient Theorem (high), GSPO (med), Limit of RLVR (pass@k) (med).

**Collection members not on the roadmap** (research-stage reasons):

- `gspo`: A research core paper (GSPO) that is not on the roadmap; see the curation notes.
- `glm-5`: A model technical report ('GLM-5: from Vibe Coding to Agentic Engineering', Feb 2026; Semantic Scholar shows 592 citations, 51 influential) describing asynchronous RL infrastructure and asynchronous agent RL algorithms. It applies this roadmap's methods at scale, and no evidence was found that others have adopted a specific algorithm from it. Better as a case-study reading than a node.
- `sdpo`: 'Reinforcement Learning via Self-Distillation' (SDPO; ICML 2026, PMLR v306) has early traction: 332 citations, 73 influential. Its update distills the model's own feedback-conditioned predictions into the policy rather than using a policy-gradient estimator, which places it in the on-policy distillation line of the post-training collection (GKD is on its roadmap; OPSD is a research core paper that is not). It is also too new to judge.
- `pkpo`: The pass@k objective is in scope, and the LRM survey cites PKPO as the unbiased pass@k estimator. But PKPO (NeurIPS 2025) has 93 citations (16 influential), is not offered in TRL, and has no other evidence of adoption, and arXiv 2511.16231 argues pass@k is a diagnostic rather than an objective. It would hang off Limit of RLVR, a research core paper that is not on the roadmap, if the pass@k line matures.
- `diffusion-nft`: Online RL for diffusion models through the forward process (ICLR 2026 Oral). This is diffusion-model RL rather than policy-gradient RL for language models, so it is outside this roadmap's scope; it is also a member of the Diffusion & Flow collection.

**Borderline candidates** (research-stage reasons):

- Learning to Summarize from Human Feedback (Stiennon et al.) (2020, arXiv 2009.01325): NeurIPS 2020, with 3,575 Semantic Scholar citations (350 influential) against Ziegler's 2,790 (213). InstructGPT follows it 'once again' for both the reward-model and PPO stages and names it with Christiano 2017 as the RLHF reference. Raschka credits it with 'the popular RLHF three-step procedure' that InstructGPT used, and the LRM survey cites it but not Ziegler. It stays borderline because it is a scaled-up refinement, by the same group, of the Ziegler 2019 recipe (reward model + PPO + KL). It would replace Ziegler if the roadmap preferred impact over origin.
- RLOO / Back to Basics (Ahmadian et al.) (2024, arXiv 2402.14740): ACL 2024. Revived REINFORCE-style optimization for RLHF with a leave-one-out baseline (from Kool et al. 2019). Adoption is real: RLOOTrainer is one of TRL's main trainers and verl supports it. The RLHF book presents RLOO as a main algorithm and shows that Dr. GRPO's advantage equals RLOO's up to a constant. It appeared three weeks after GRPO, which became the base of R1 and its successors, so including both would put two near-identical group baselines on the roadmap. Its citation count could not be retrieved.
- Truncated Importance Sampling for rollout-training mismatch (Yao et al.) (2025): The primary sources are a Notion blog (Aug 2025) and an OPT 2025 workshop paper ('On the Rollout-Training Mismatch in Modern RL Systems'); the technique itself is Ionides (2008). The RLHF book calls TIS 'a crucial tool' for asynchronous LLM RL, the LRM survey cites it, and TRL turns importance-sampling correction on by default. But TRL's default mode is now sequence-level masking (MIS/IcePop-style), so the specific fix is already being superseded. It stays borderline until one off-policy correction settles.
- CISPO (MiniMax-M1) (2025, arXiv 2506.13585): Clips and stops the gradient through the importance weight instead of clipping the objective, so every token keeps a gradient. It has its own section in the RLHF book, is a TRL loss type, and was adopted by Meta's ScaleRL recipe (TRL documents epsilon_high=5.0 for it). The MiniMax-M1 report has 240 citations (27 influential), counting the whole model report. It is the strongest alternative to GSPO (not on the roadmap) as the 2025 ratio-control method.
- Pass@k Training (Chen et al.) (2025, arXiv 2508.10751): Uses pass@k as the training reward to balance exploration and exploitation. It cites Limit of RLVR as motivation and is listed in the LRM survey. It has 138 citations (28 influential), more than PKPO's 93, but no library default, and arXiv 2511.16231 argues pass@k is a diagnostic rather than an objective.
- Natural Policy Gradient (Kakade) (2001): Listed in Spinning Up's classic papers, and covered in CS285's advanced policy-gradient lecture (which cites Peters & Schaal 2008 for it). TRPO says its update 'is similar to natural policy gradient methods' and recovers NPG as a special case. Lilian Weng mentions it only in passing. TRPO is the node that made trust regions practical for deep networks, and no LLM RL method uses NPG directly.
- Conservative Policy Iteration (Kakade & Langford) (2002): TRPO's main theorem extends this paper's policy-improvement bound, and it is listed in Spinning Up's classic papers. TRPO already carries the idea. It would be the classical trust-region origin above TRPO, but it has no arXiv version.
- A3C / A2C (2016, arXiv 1602.01783): Semantic Scholar shows 10,441 citations (1,343 influential), and it is on the Spinning Up and Lilian Weng lists. PPO compares against A2C and A2C with a trust region. Its asynchronous design was superseded, and the LLM lineage runs through PPO rather than A3C.
- IMPALA / V-trace (2018, arXiv 1802.01561): The classical truncated-importance-weight correction for decoupled actor-learner RL (covered by Lilian Weng). The LLM-era corrections do not build on it: the TIS paper cites Ionides 2008, and TRL cites Retrace (Munos et al. 2016).
- Tulu 3 (RLVR) (2024, arXiv 2411.15124): Coined the name 'Reinforcement Learning with Verifiable Rewards'. The paradigm was popularized by DeepSeek-R1, and Tulu 3's RL stage uses PPO, so it adds no new algorithm.
- ScaleRL (The Art of Scaling RL Compute for LLMs) (2025, arXiv 2510.13786): A large compute-scaling study by Meta that proposes the ScaleRL recipe (CISPO loss among other choices); TRL documents how to reproduce it. It has 103 citations (13 influential) in a year and is an empirical recipe study rather than a new method.

**Rejected notable candidates** (research-stage reasons):

- DQN: Value-based deep RL, outside the policy-gradient scope. It is on syllabi (CS224R off-policy lecture) but is not an ancestor of LLM RL objectives.
- DDPG / TD3 / SAC: Off-policy continuous-control actor-critic methods (Spinning Up key papers). No LLM RL method builds on them.
- DPO: Offline preference optimization, not policy-gradient RL. It belongs on the post-training roadmap.
- OpenAI o1: A turning point (the LRM survey pairs it with R1 as a milestone), but it has no technical paper describing the method (blog and system card only), so there is no citable algorithmic lineage.
- Kimi k1.5: A detailed report released alongside R1 that uses online policy mirror descent rather than GRPO. Lambert notes it lacked R1's open weights and viral moment, and the later GRPO-line papers build on R1, not on k1.5.
- ReMax / REINFORCE++: Incremental critic-free baselines. They are supported in verl and listed in the LRM survey's algorithm table, but they are not field-defining.
- VAPO / Lite PPO / GMPO / SAPO: Incremental 2025 PPO/GRPO variants, each adjusting one component of the core nodes.
- ACER / ACKTR: Listed by Spinning Up but superseded by PPO; no descendant in the LLM line.
- Constitutional AI / Anthropic HH-RLHF: Important for alignment data and AI feedback, but they apply the InstructGPT-style PPO recipe rather than changing the RL algorithm.
- STaR (Self-Taught Reasoner): An early reasoning bootstrapping method that DeepSeek-R1 cites, but it is filtered self-training (expert iteration), not policy-gradient RL.
- Process reward models (Let's Verify Step by Step): A reward-modeling contribution (process supervision), not a policy-optimization method. The R1 line uses rule-based outcome rewards instead.
- Open-Reasoner-Zero / SimpleRL-Zoo: Open reproductions of R1-Zero-style training. Lambert describes SimpleRL-Zoo as 'a more comprehensive version of OpenReasonerZero'. Useful replications, not new methods.
- Clip-Cov / KL-Cov (entropy mechanism of RL for reasoning LMs): Entropy-collapse analysis with covariance-based clipping. verl supports it and the LRM survey lists it, but it is one of many entropy-control tweaks on GRPO/PPO.

**Curation notes:** Selection rule: high-confidence papers first, then papers that start or connect a line. The research kept 14 papers: 10 high, 4 medium. Fine-Tuning LMs from Human Preferences (medium) stays because it connects Christiano 2017 to InstructGPT. The other medium papers, Dr. GRPO, GSPO and Limit of RLVR, are all leaves. Of the three medium leaves, only the earliest, Dr. GRPO (March 2025), is shown to keep the map small; GSPO and Limit of RLVR are left out. GRPO → DAPO and GRPO → Dr. GRPO are not drawn, because DeepSeek-R1 already connects them. PPO → Ziegler and InstructGPT → GRPO are not drawn, because each would turn another kept arrow into a skip. PPO → InstructGPT and PPO → GRPO are kept, since GRPO is a PPO variant. REINFORCE and the Policy Gradient Theorem are left out because every roadmap paper must have an arXiv version (the panel shows its arXiv card). TRPO is the start (earliest paper, first row, no incoming arrow). TRPO → GAE is not drawn because it would turn TRPO → PPO into a skip.

## Diffusion & Flow

Continuous diffusion and flow models: the 2015 diffusion and 2019 score-based roots, DDPM and the score-SDE view, faster deterministic sampling (DDIM), guidance (ADM, classifier-free guidance), latent diffusion, the EDM design space, flow matching and rectified flow, and consistency models for few-step generation.

| Paper | Year | Venue | Exercise | Confidence |
| --- | --- | --- | --- | --- |
| [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) | 2015 | ICML 2015 | grey | high |
| [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) | 2019 | NeurIPS 2019 | grey | high |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | NeurIPS 2020 | `ddpm` | high |
| [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | 2020 | ICLR 2021 | `ddim` | high |
| [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | 2020 | ICLR 2021 | grey | high |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | 2021 | NeurIPS 2021 | grey | high |
| [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 2021 | CVPR 2022 | grey | high |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 2021 | NeurIPS 2021 Workshop | grey | high |
| [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) | 2022 | NeurIPS 2022 | grey | high |
| [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) | 2022 | ICLR 2023 | grey | high |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 2022 | ICLR 2023 | `flow-matching` | high |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023 | ICML 2023 | `consistency` | high |

**Evidence per paper** (type, linked source, shortened note):

- **Sohl-Dickstein** (2015): Introduced diffusion models: slowly add noise to data, then learn to reverse the process step by step.
  - survey · [Flow Matching Guide and Code (Lipman et al., 2024)](https://arxiv.org/abs/2412.06264): 'Diffusion Models were originally introduced as discrete time Gaussian processes (Sohl-Dickstein et al., 2015; Ho et al., 2020)'; verified in the PDF text.
  - survey · [Yang et al., Diffusion Models: A Comprehensive Survey of Methods and Applications (ACM CSUR)](https://arxiv.org/abs/2209.00796): Cites Sohl-Dickstein 2015 with Ho 2020 for DDPMs, the first of its 'three predominant formulations'; verified in the PDF text.
  - community · [Lilian Weng, What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): Named first among the diffusion-based models the post builds on, with NCSN and DDPM; the Swiss-roll figure comes from this paper.
  - community · [Yang Song blog, Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/): 'Diffusion probabilistic models are hierachical latent variable models first proposed by Jascha and his colleagues in 2015'.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1503.03585?fields=title,year,citationCount,influentialCitationCount): 10,991 citations, 547 influential (not re-fetched).
- **NCSN** (2019): Trained one network to estimate the score (gradient of log density) at many noise levels and sampled with Langevin dynamics.
  - survey · [Yang et al., Diffusion Models: A Comprehensive Survey (ACM CSUR)](https://arxiv.org/abs/2209.00796): Score-based generative models (SGMs) are the second of the survey's 'three predominant formulations'.
  - community · [Yang Song blog](https://yang-song.net/blog/2021/score/): Presents NCSN as the foundation of score-based generative modeling, with Score SDE as its continuous-time generalization.
  - community · [Lilian Weng, What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): Has its own subsection, 'Connection with noise-conditioned score networks (NCSN)'.
  - community · [EleutherAI Diffusion Reading Group](https://github.com/tmabraham/diffusion_reading_group): Session 2 is 'Score-based generative modeling'.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1907.05600?fields=title,year,citationCount,influentialCitationCount): 6,122 citations, 513 influential.
- **DDPM** (2020): Made diffusion work for high-quality images with a simple noise-prediction loss; its recipe became the standard.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2006.11239?fields=title,year,citationCount,influentialCitationCount): 36,546 citations, 5,037 influential; the most-cited paper in this roadmap.
  - survey · [Yang et al., Diffusion Models: A Comprehensive Survey (ACM CSUR)](https://arxiv.org/abs/2209.00796): DDPMs are the first of the survey's 'three predominant formulations'.
  - community · [OPIG blog, A timeline of sampling methods of diffusion models (2026)](https://www.blopig.com/blog/2026/05/a-timeline-of-sampling-methods-of-diffusion-models/): Traces the field 'through four foundational works: DDPM, DDIM, the score-based SDE unification, and EDM'.
  - community · [Lilian Weng, What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): DDPM is the central reference for the post's derivations and training objective.
  - adoption · [Hugging Face diffusers DDIMScheduler docs](https://huggingface.co/docs/diffusers/api/schedulers/ddim): 'DDIMScheduler extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs)'; eta=1 corresponds to DDPM.
- **DDIM** (2020): A deterministic sampler that shares DDPM's training but needs far fewer steps, and maps noise to images invertibly.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2010.02502?fields=title,year,citationCount,influentialCitationCount): 14,001 citations, 1,857 influential.
  - adoption · [Hugging Face diffusers DDIMScheduler docs](https://huggingface.co/docs/diffusers/api/schedulers/ddim): Core diffusers scheduler; 'A value of 0 corresponds to DDIM (deterministic)'.
  - adoption · [CompVis/stable-diffusion README](https://github.com/CompVis/stable-diffusion): The reference txt2img script exposes DDIM sampling (--ddim_steps, --ddim_eta); its default sampler is PLMS.
  - community · [OPIG blog, A timeline of sampling methods of diffusion models](https://www.blopig.com/blog/2026/05/a-timeline-of-sampling-methods-of-diffusion-models/): One of the 'four foundational works'.
  - community · [Sander Dieleman, Learning the integral of a diffusion model (flow maps)](https://sander.ai/2026/05/06/flow-maps.html): Cites DDIM (ref. 6) for the deterministic update rule that flow-map methods start from.
- **Score SDE** (2020): Unified score models and DDPM as stochastic differential equations and introduced the probability flow ODE.
  - survey · [Yang et al., Diffusion Models: A Comprehensive Survey (ACM CSUR)](https://arxiv.org/abs/2209.00796): Score SDEs are the third of the survey's 'three predominant formulations'.
  - survey · [Flow Matching Guide and Code (Lipman et al., 2024)](https://arxiv.org/abs/2412.06264): Diffusion models were 'later formulated in terms of continuous time Stochastic Differential Equations (SDEs) (Song et al., 2021)'.
  - community · [Yang Song blog](https://yang-song.net/blog/2021/score/): 'This work bridges both score-based generative modeling and diffusion probabilistic modeling into a unified framework.'
  - community · [OPIG blog timeline](https://www.blopig.com/blog/2026/05/a-timeline-of-sampling-methods-of-diffusion-models/): 'The score-based SDE unification' is one of its four foundational works.
  - syllabus · [MIT 6.S184 (2026)](https://diffusion.csail.mit.edu/2026/): The course is built on the ODE/SDE view; lecture 3-A covers score functions, denoising score matching and SDE sampling.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2011.13456?fields=title,year,citationCount,influentialCitationCount): 12,622 citations, 2,045 influential.
- **ADM** (2021): Showed diffusion models beating GANs on ImageNet, using a better UNet and classifier guidance.
  - citations · [Paper Digest, Most Influential NeurIPS 2021 Papers (2026-03 edition; seen in search results, page behind a captcha)](https://www.paperdigest.org/certificate/21a41c2dd2aa2f54): Ranked No. 1 most influential NeurIPS 2021 paper. Semantic Scholar (not re-fetched): 13,400 citations, 1,204 influential.
  - community · [Sander Dieleman, Guidance: a cheat code for diffusion models](https://sander.ai/2022/05/26/guidance.html): Dhariwal & Nichol 'really drove this point home' that classifier guidance substantially improves sample quality.
  - community · [Lilian Weng, What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): 'Classifier Guided Diffusion' opens the 'Conditioned Generation' section.
  - community · [EleutherAI Diffusion Reading Group](https://github.com/tmabraham/diffusion_reading_group): Session 5 covers IDDPM, ADM and classifier-free guidance.
  - other · [NeurIPS 2021 spotlight listing (seen in search results)](https://neurips.cc/virtual/2021/spotlight/30543): Presented as a NeurIPS 2021 spotlight.
- **Latent Diffusion** (2021): Ran diffusion in an autoencoder's latent space, making high-resolution text-to-image affordable; the basis of Stable Diffusion.
  - adoption · [CompVis/stable-diffusion README](https://github.com/CompVis/stable-diffusion): 'Stable Diffusion is a latent text-to-image diffusion model'.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2112.10752?fields=title,year,citationCount,influentialCitationCount): 28,275 citations and 5,842 influential, the most influential citations of any paper in this roadmap.
  - community · [Lilian Weng, What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/): Has its own 'Latent Variable Space' subsection.
  - community · [EleutherAI Diffusion Reading Group](https://github.com/tmabraham/diffusion_reading_group): Session 13 covers Latent Diffusion and Stable Diffusion, and session 14 is a Q&A with Robin Rombach.
  - syllabus · [MIT 6.S184 (2026)](https://diffusion.csail.mit.edu/2026/): Students build a latent diffusion model from scratch; lecture 4 covers latent spaces.
- **CFG** (2021): Guides generation without a separate classifier by mixing conditional and unconditional predictions; now the default.
  - adoption · [CompVis/stable-diffusion README](https://github.com/CompVis/stable-diffusion): Default '--scale 7.5' unconditional guidance; the sd-v1-3/v1-4 checkpoints use '10% dropping of the text-conditioning to improve classifier-free guidance …
  - adoption · [Imagen Video (Ho et al., 2022)](https://arxiv.org/abs/2210.02303): 'We apply progressive distillation to our video models with classifier-free guidance for fast, high quality sampling.'
  - community · [Sander Dieleman, Guidance: a cheat code for diffusion models](https://sander.ai/2022/05/26/guidance.html): 'A real game changer'; 'an essential component of OpenAI's DALL·E 2 and Google's Imagen'; GLIDE used it 'to great effect'.
  - syllabus · [MIT 6.S184 (2026)](https://diffusion.csail.mit.edu/2026/): Lecture 3-B is titled 'Classifier-free Guidance'.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2207.12598?fields=title,year,citationCount,influentialCitationCount): 7,480 citations, 1,079 influential.
- **EDM** (2022): Separated diffusion design choices (noise levels, preconditioning, loss weighting, sampler) into one framework; a default setup since.
  - other · [NeurIPS 2022 Awards announcement](https://blog.neurips.cc/2022/11/21/announcing-the-neurips-2022-awards/): One of 13 Outstanding Main Track Papers.
  - community · [Sander Dieleman, The paradox of diffusion distillation](https://sander.ai/2024/02/28/paradox.html): Calls Karras et al. 2022 an 'instant classic' and uses it as the reference study of sampler design.
  - community · [OPIG blog, A timeline of sampling methods of diffusion models](https://www.blopig.com/blog/2026/05/a-timeline-of-sampling-methods-of-diffusion-models/): One of 'four foundational works'; says EDM 'has become the default in modern image-diffusion and protein-diffusion codebases'.
  - community · [EleutherAI Diffusion Reading Group](https://github.com/tmabraham/diffusion_reading_group): Session 8 covers Elucidating Design Spaces (EDM), and session 9 is a DDPM and EDM code walk-through.
  - syllabus · [Technion CS 236759 Generative AI - Diffusion Models, list of papers (2025)](https://elad.cs.technion.ac.il/wp-content/uploads/2025/03/236759-List-of-Papers.pdf): Entry #100, the first paper under 'Architecture and Training'.
  - syllabus · [UIUC ECE 598ZZ (Fall 2025)](https://courses.grainger.illinois.edu/ece598zz/fa2025/): Lecture 9 is titled 'Design space of diffusion models'.
  - adoption · [Hugging Face diffusers HeunDiscreteScheduler docs](https://huggingface.co/docs/diffusers/main/api/schedulers/heun): 'The Heun scheduler (Algorithm 1) is from the Elucidating the Design Space of Diffusion-Based Generative Models paper'; most schedulers expose …
  - citations · [Paper Digest, Most Influential NeurIPS 2022 Papers (seen in search results, page behind a captcha)](https://www.paperdigest.org/certificate/93234f77c2fd3454): Ranked No. 8 for NeurIPS 2022. Semantic Scholar (not re-fetched): 4,038 citations, 801 influential.
- **Rectified Flow** (2022): Learns straight paths between noise and data, which makes few-step sampling easier; Stable Diffusion 3 and FLUX train with its formulation.
  - adoption · [Stable Diffusion 3 paper (Esser et al., 2024)](https://arxiv.org/abs/2403.03206): 'Scaling Rectified Flow Transformers'; reports 'superior performance of this approach compared to established diffusion formulations'.
  - adoption · [FLUX.1 [dev] model card](https://huggingface.co/black-forest-labs/FLUX.1-dev): 'FLUX.1 [dev] is a 12 billion parameter rectified flow transformer'.
  - community · [Sander Dieleman, The paradox of diffusion distillation](https://sander.ai/2024/02/28/paradox.html): Discusses reflow as a distinct path-straightening approach to fast sampling.
  - syllabus · [UIUC ECE 598ZZ (Fall 2025)](https://courses.grainger.illinois.edu/ece598zz/fa2025/): Lecture 12 is titled 'Rectified flow and flow matching'.
  - syllabus · [Technion CS 236759 list of papers](https://elad.cs.technion.ac.il/wp-content/uploads/2025/03/236759-List-of-Papers.pdf): Entry #300, first under 'Distillation and Acceleration'.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2209.03003?fields=title,year,citationCount,influentialCitationCount): 4,391 citations, 449 influential.
- **Flow Matching** (2022): Trains generators by regressing the velocity of a chosen probability path, with diffusion paths as special cases.
  - adoption · [Movie Gen (Polyak et al., 2024)](https://arxiv.org/abs/2410.13720): 'We use the Flow Matching training objective (Lipman et al., 2023) to train our generative model'; verified in the PDF text.
  - survey · [Flow Matching Guide and Code (Lipman et al., 2024)](https://arxiv.org/abs/2412.06264): Lists large-scale uses: images (Esser et al. 2024), video (Polyak et al. 2024), speech, audio, proteins and robotics.
  - syllabus · [MIT 6.S184 (2026)](https://diffusion.csail.mit.edu/2026/): Lecture 2 is titled 'Flow Matching'.
  - syllabus · [Technion CS 236759 list of papers](https://elad.cs.technion.ac.il/wp-content/uploads/2025/03/236759-List-of-Papers.pdf): Entry #305.
  - community · [EleutherAI Diffusion Reading Group](https://github.com/tmabraham/diffusion_reading_group): Sessions 16-17 cover Flow Matching, and session 19 covers Conditional Flow Matching.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2210.02747?fields=title,year,citationCount,influentialCitationCount): 7,095 citations, 954 influential.
- **Consistency** (2023): Learns to map any point on a sampling trajectory straight to its endpoint, enabling one- or few-step generation.
  - community · [Sander Dieleman, Learning the integral of a diffusion model (flow maps)](https://sander.ai/2026/05/06/flow-maps.html): Says consistency models 'set off something of a chain reaction' of flow-map methods.
  - other · [MeanFlow (Geng et al., 2025), introduction](https://arxiv.org/abs/2505.13447): 'Pioneering this direction, Consistency Models introduce a consistency constraint ...' (verified in the PDF text).
  - adoption · [sCM: Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (Lu & Song, ICLR 2025 Oral)](https://arxiv.org/abs/2410.11081): Scales consistency models to 1.5B parameters; 1.88 FID on ImageNet 512x512 with two steps.
  - syllabus · [UIUC ECE 598ZZ (Fall 2025) and Technion CS 236759](https://courses.grainger.illinois.edu/ece598zz/fa2025/): UIUC lecture 10 is titled 'Consistency models'; the Technion paper list includes Consistency Models as entry #301.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2303.01469?fields=title,year,citationCount,influentialCitationCount): 2,351 citations, 314 influential; the latent follow-up LCM adds 995.

**Arrows** (the relation sentence shown on the site, then its source and quote):

- Sohl-Dickstein → DDPM: DDPM took Sohl-Dickstein's diffusion models and made them generate high-quality images. ([source](https://arxiv.org/abs/2006.11239): “We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from …”)
- NCSN → DDPM: DDPM showed its noise-prediction training is equivalent to NCSN's score matching. ([source](https://ar5iv.labs.arxiv.org/html/2006.11239): “we show that a certain parameterization of diffusion models reveals an equivalence with denoising score matching over multiple noise levels during training and …”)
- DDPM → DDIM: DDIM kept DDPM's training as is and swapped in a sampler that needs far fewer steps. ([source](https://arxiv.org/abs/2010.02502): “To accelerate sampling, we present denoising diffusion implicit models (DDIMs), a more efficient class of iterative implicit probabilistic models with the same …”)
- DDPM → Score SDE: Score SDE rewrote DDPM's step-by-step noising as a continuous-time SDE. ([source](https://ar5iv.labs.arxiv.org/html/2011.13456): “The methods of SMLD and DDPM can be amalgamated into our framework as discretizations of two separate SDEs.”)
- Score SDE → ADM: ADM built classifier guidance on Score SDE's idea of steering samples with a classifier. ([source](https://ar5iv.labs.arxiv.org/html/2105.05233): “Sohl-Dickstein et al. 2015 [56] and Song et al. 2020b [60] show one way to achieve this, wherein a pre-trained diffusion model can be conditioned using the …”)
- ADM → CFG: CFG removed ADM's extra classifier by mixing conditional and unconditional predictions. ([source](https://ar5iv.labs.arxiv.org/html/2207.12598): “Dhariwal & Nichol 2021 proposed classifier guidance, a technique to boost the sample quality of a diffusion model using an extra trained classifier. ... We are …”)
- DDPM → Latent Diffusion: Latent Diffusion ran DDPM's noise-prediction training on autoencoder latents, not pixels. ([source](https://arxiv.org/pdf/2112.10752): “The best synthesis quality is usually achieved when a reweighted objective [30] is used for training. ... we apply them in the latent space of powerful …”)
- Score SDE → EDM: EDM took Score SDE's models and retuned their noise schedule, sampler and training. ([source](https://arxiv.org/pdf/2206.00364): ““DDPM++ cont. (VP)” and “NCSN++ cont. (VE)” models by Song et al. [49] ... VP and VE benefit considerably from switching away from their original schedules. …”)
- Score SDE → Flow Matching: Flow Matching generalized Score SDE's routes for turning noise into data. ([source](https://ar5iv.labs.arxiv.org/html/2210.02747): “Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples—which subsumes existing …”)
- Score SDE → Rectified Flow: Rectified Flow replaced Score SDE's curved ODE paths with straight noise-to-data lines. ([source](https://ar5iv.labs.arxiv.org/html/2209.03003): “In [73], three types of PF-ODEs are derived from three types of SDEs learned as score-based generative models ... the trajectories of VP ODE and sub-VP ODE are …”)
- EDM → Consistency: Consistency models adopted EDM's setup and distilled EDM models into one-step generators. ([source](https://ar5iv.labs.arxiv.org/html/2303.01469): “We hereafter adopt the settings in Karras et al. 2022 ... In both consistency distillation and progressive distillation, we distill EDMs (Karras et al. 2022).”)

**Research core papers not on the roadmap:** Progressive Distillation (med), DMD (med), MeanFlow (med).

**Collection members not on the roadmap** (research-stage reasons):

- `progressive-distillation`: A research core paper (Progressive Distillation) that is not on the roadmap; see the curation notes.
- `dmd`: A research core paper (DMD) that is not on the roadmap; see the curation notes.
- `dmd2`: Refinement of DMD: it removes the regression loss, adds a GAN loss and two-time-scale updates, and scales to SDXL. Well cited and used in video distillation recipes, but DMD, a research core paper that is not on the roadmap, opened distribution-matching distillation. DMD2 makes a natural follow-up exercise.
- `meanflow`: A research core paper (MeanFlow) that is not on the roadmap; see the curation notes.
- `jit`: November 2025 Li & He tech report arguing that denoisers should predict clean images (x-prediction) with plain pixel-space Transformers. It draws expert attention but is too recent for survey or adoption evidence; borderline, worth revisiting in 2027.
- `repa`: ICLR 2025 Oral training-acceleration technique that aligns diffusion-transformer features with pretrained self-supervised encoders. Influential, but it changes training efficiency rather than the diffusion or flow formulation, and it also fits a representation-learning roadmap (it is on the Visual Generation roadmap).
- `block-diffusion`: ICLR 2025 Oral on discrete diffusion language models that interpolate with autoregressive decoding. Outside this roadmap's continuous diffusion and flow scope; belongs in a discrete-diffusion or diffusion-LM collection.
- `svdquant`: ICLR 2025 Spotlight on 4-bit post-training quantization of diffusion models. A systems and efficiency method that does not change how diffusion models are trained or sampled; fits a quantization or efficiency roadmap.
- `daps`: Inverse-problem solver based on decoupled noise annealing. Diffusion inverse problems are an application subfield, and within it DPS opened the line, not DAPS.
- `diffusion-nft`: ICLR 2026 Oral on online RL fine-tuning of diffusion and flow models through the forward process. It is RL fine-tuning rather than a change to the diffusion or flow formulation; it is also a member of the RL collection.
- `fd-loss`: April 2026 paper that post-trains generators with a Frechet-distance loss in representation space. Too new for evidence of field-level impact.

**Borderline candidates** (research-stage reasons):

- DiT (Scalable Diffusion Models with Transformers) (2022, arXiv 2212.09748): Very strong case: 7,763 citations and 889 influential (Semantic Scholar, not re-fetched). It is covered in MIT 6.S184 lecture 4, Lilian Weng's 'Model Architecture' section and EleutherAI reading group session 27, and it is the backbone family behind SD3, FLUX and modern video models. Kept out because it changes the network architecture, not the diffusion or flow formulation, and the field scope does not list architectures. It belongs on an architecture roadmap and could be cross-listed here as a grey node.
- Improved DDPM (2021, arXiv 2102.09672): 6,017 citations and 463 influential (Semantic Scholar, not re-fetched). Introduced the cosine schedule and learned variances, which ADM builds on directly; covered in Weng's post and EleutherAI session 5. A refinement of DDPM rather than a new line.
- Stochastic Interpolants (2022, arXiv 2209.15571): Concurrent co-founder of flow matching; the Flow Matching paper names it with Rectified Flow as concurrent work, and the Movie Gen report cites it. Far less cited (1,148 citations, 50 influential, Semantic Scholar, not re-fetched). Flow Matching and Rectified Flow already represent this moment.
- Stable Diffusion 3 (Scaling Rectified Flow Transformers) (2024, arXiv 2403.03206): 5,009 citations and 861 influential (Semantic Scholar, not re-fetched); the paper that established rectified flow at scale. Used here as adoption evidence for Rectified Flow. As a system paper it combines existing ideas rather than introducing one.
- DPM-Solver (2022, arXiv 2206.00927): 2,693 citations (Semantic Scholar, not re-fetched); entry #200 on the Technion paper list. A widely used training-free high-order ODE sampler. An important refinement of the DDIM and probability-flow sampling line rather than a line opener.
- Latent Consistency Models (LCM) (2023, arXiv 2310.04378): 995 citations (Semantic Scholar, not re-fetched); brought consistency distillation to Stable Diffusion (LCM-LoRA). It is the main adoption route for consistency models, but a direct application of them.
- Adversarial Diffusion Distillation (SDXL Turbo) (2023, arXiv 2311.17042): 928 citations and 133 influential (Semantic Scholar, not re-fetched). Shipped as SDXL Turbo and started an adversarial-distillation branch; Dieleman describes it as SDS combined with an adversarial loss. The main alternative to DMD as a distillation candidate; neither is on the roadmap.
- sCM (continuous-time consistency models) (2024, arXiv 2410.11081): ICLR 2025 Oral (re-fetched); scales consistency models to 1.5B parameters with 1.88 FID on ImageNet 512x512 in two steps, and its TrigFlow formulation unifies EDM and flow matching. Used here as adoption evidence for Consistency Models; a refinement of them.
- Shortcut Models (2024, arXiv 2410.12557): ICLR 2025 (cited as such by MeanFlow), 458 citations (Semantic Scholar, not re-fetched). Conditions on step size for one-step generation. A MeanFlow predecessor in the flow-map family that MeanFlow's results surpass.
- Inductive Moment Matching (IMM) (2025, arXiv 2503.07565): It is entry #304 on the Technion paper list, Dieleman's flow-maps post lists it among distributional distillation methods, and MeanFlow benchmarks against it. Same one-step-from-scratch family as MeanFlow with lower uptake (125 citations, not re-fetched).
- DMD2 (2024, arXiv 2405.14867): Catalog member. Drops DMD's regression loss and adds a GAN loss with two-time-scale updates (772 citations, 138 influential, Semantic Scholar, not re-fetched). Widely used in practice, but a refinement of DMD.
- REPA (2024, arXiv 2410.06940): Catalog member; ICLR 2025 Oral (re-fetched), 796 citations and 170 influential (Semantic Scholar, not re-fetched). Speeds up DiT training by aligning hidden features with pretrained self-supervised encoders. Influential, but a training-efficiency technique rather than a change to the diffusion or flow formulation.
- JiT (Back to Basics: Let Denoising Generative Models Denoise) (2025, arXiv 2511.13720): Catalog member; Li & He tech report (Nov 2025) arguing for clean-image (x) prediction with plain pixel-space Transformers. Too recent to show survey or adoption evidence; worth revisiting in 2027.

**Rejected notable candidates** (research-stage reasons):

- GLIDE / DALL-E 2 (unCLIP) / Imagen: Highly cited text-to-image systems (Imagen is a NeurIPS 2022 Outstanding Paper). They integrate guidance, cascades and pixel or latent diffusion, and their method contributions are captured by the CFG and LDM nodes.
- ControlNet: A conditioning adapter for pretrained text-to-image models (8,006 citations, Semantic Scholar, not re-fetched). It belongs to a controllable-generation or editing roadmap, not the modeling and sampling core.
- Diffusion Posterior Sampling (DPS): Opened diffusion-based inverse-problem solving, the line DAPS belongs to (2,040 citations, Semantic Scholar, not re-fetched). An application subfield outside this roadmap's scope.
- Score matching (Hyvarinen 2005) and denoising score matching (Vincent 2011): Mathematical precursors that predate deep diffusion generative modeling and have no arXiv versions. NCSN represents their entry into the field.
- Denoising Student (Luhman & Luhman 2021): The distillation survey says it 'may be the first to distill pre-trained DMs', and Progressive Distillation calls it the closest prior work. It was superseded by progressive distillation, which the blogs, the reading group and Imagen Video feature.
- Diff-Instruct (Luo et al. 2023): Earlier (May 2023) one-step distillation via an integral KL divergence with an auxiliary diffusion model. Dieleman notes DMD 'arrives at a very similar formulation from a different angle'. DMD, a research core paper that is not on the roadmap, is preferred because DMD and DMD2 are the variants adopted downstream (CausVid, Self Forcing).
- DreamFusion SDS / ProlificDreamer VSD: DMD says it 'draws upon inspiration and insights from VSD', so VSD is DMD's direct conceptual parent. Their target, test-time optimization of 3D representations, is outside this roadmap's scope.
- Variational Diffusion Models (Kingma et al. 2021): Important for the SNR and continuous-time likelihood view, and covered in EleutherAI reading group session 7. No survey, blog or syllabus checked presents it as a turning point, and its practical design lessons are largely absorbed by Score SDE and EDM.
- Discrete diffusion (D3PM, MDLM, SEDD, Block Diffusion): Diffusion over discrete tokens, mainly for language. It uses different math and has a different community, and deserves its own roadmap (MIT 6.S184 gives it a separate lecture).
- InstaFlow: Applies reflow to Stable Diffusion for one-step text-to-image generation; a direct application of Rectified Flow.

**Curation notes:** Selection rule: high-confidence papers first, then papers that start or connect a line. Exactly 12 of the 15 researched papers were rated high. Progressive Distillation, DMD and MeanFlow (all medium) are not on the roadmap (listed above). Three arrows are not drawn because a longer path already connects the same papers: NCSN → Score SDE and DDPM → ADM (both via DDPM and Score SDE), and Score SDE → Consistency (via EDM). The start node is DDPM, centered on the canvas at column 1.5.

## Post-training

Turning a pretrained language model into an assistant: RLHF from human preferences to InstructGPT and DPO, AI feedback (Constitutional AI), instruction tuning (FLAN), parameter-efficient fine-tuning (Adapters, LoRA, QLoRA), and knowledge distillation up to on-policy distillation (GKD).

| Paper | Year | Venue | Exercise | Confidence |
| --- | --- | --- | --- | --- |
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | 2014 | NeurIPS 2014 Workshop | grey | high |
| [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | NeurIPS 2017 | grey | high |
| [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) | 2019 | ICML 2019 | grey | med |
| [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) | 2019 | arXiv | grey | high |
| [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) | 2020 | NeurIPS 2020 | grey | med |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | 2021 | ICLR 2022 | `lora` | high |
| [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) | 2021 | ICLR 2022 | grey | high |
| [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | 2022 | NeurIPS 2022 | grey | high |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | 2022 | arXiv | grey | high |
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | 2023 | NeurIPS 2023 | `dpo` | high |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | NeurIPS 2023 | `qlora` | high |
| [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649) | 2023 | ICLR 2024 | `gkd` | high |

**Evidence per paper** (type, linked source, shortened note):

- **Distillation** (2014): Introduced knowledge distillation: train a small student to match a large teacher's softened output distribution.
  - survey · [Song & Zheng, A Survey of On-Policy Distillation for Large Language Models](https://arxiv.org/html/2604.00626v3): Knowledge distillation was 'first formalized by Hinton et al. (2015)'. The survey treats it as the precursor of off-policy and on-policy LLM distillation.
  - adoption · [Gemma 2 technical report](https://arxiv.org/abs/2408.00118): 'We also train the 2B and 9B models with knowledge distillation (Hinton et al., 2015) instead of next token prediction.'
  - other · [GKD and MiniLLM papers](https://arxiv.org/abs/2306.13649): Both open from Hinton-style KD. GKD uses 'Supervised KD (Hinton et al., 2015)' as a baseline, and MiniLLM starts from 'knowledge distillation (KD; HVD15)'.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1503.02531?fields=title,year,venue,citationCount,influentialCitationCount): About 26,891 citations, 2,967 influential.
- **Christiano RLHF** (2017): Learned a reward model from human comparisons of two behaviors and used it instead of a hand-written reward for RL.
  - survey · [Lambert, RLHF Book, ch. Related Works](https://rlhfbook.com/c/02-related-works): The chapter's reference for the start of modern RLHF: 'an application of RLHF applied to preferences between trajectories of agents within Atari games'.
  - community · [Hugging Face blog 'Illustrating RLHF', Further reading](https://huggingface.co/blog/rlhf): Listed among the early RLHF papers that come before the language-model work.
  - community · [daily.dev key papers reading list](https://daily.dev/agentic-ai-hub/key-papers-reading-list.md/): Tagged 'Landmark': 'The origin of RLHF: learning a reward from human comparisons. The recipe InstructGPT later scaled to language.'
  - other · [Constitutional AI paper, Related Work](https://arxiv.org/abs/2212.08073): 'Our work can be thought of as an extension of RLHF [Christiano et al., 2017] with language models [Stiennon et al., 2020]'.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03741?fields=title,year,venue,citationCount,influentialCitationCount): About 6,200 citations, 688 influential.
- **Adapters** (2019): Froze the pretrained model and trained small inserted bottleneck modules, starting parameter-efficient fine-tuning.
  - syllabus · [Stanford CS224N (2026) 'Efficient Adaptation (Prompting + PEFT)' lecture](https://web.stanford.edu/class/cs224n/): Suggested reading, alongside LoRA.
  - survey · [Zhao et al., A Survey of Large Language Models, Sec. 5.3.1](https://arxiv.org/abs/2303.18223): Adapter tuning is one of the four representative PEFT methods it reviews (adapter, prefix, prompt tuning, LoRA).
  - other · [LoRA paper](https://arxiv.org/abs/2106.09685): 'We focus on the original design by Houlsby et al. (2019)'. Uses it as the main adapter baseline and motivates LoRA by adapter inference latency.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1902.00751?fields=title,year,venue,citationCount,influentialCitationCount): About 7,578 citations, 919 influential.
- **Ziegler RLHF** (2019): First applied a preference reward model plus PPO to a pretrained language model, with a KL penalty to the initial model.
  - survey · [Lambert, RLHF Book, ch. Related Works](https://rlhfbook.com/c/02-related-works): 'Many canonical terms, such as learning reward models, KL distances, feedback diagrams, etc., were formalized in this paper.'
  - syllabus · [Princeton COS 597G (Fall 2022), 'Training LMs with human feedback'](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/): Recommended reading, next to Stiennon et al.; InstructGPT is the required reading.
  - community · [Hugging Face blog 'Illustrating RLHF'](https://huggingface.co/blog/rlhf): First entry under 'Key papers for LMs'.
  - other · [InstructGPT and DPO papers](https://arxiv.org/abs/2305.18290): DPO: 'We review the RLHF pipeline in Ziegler et al. (and later [Stiennon, Bai, Ouyang])'. InstructGPT: 'Our methodology follows that of Ziegler et al. (2019) …
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:1909.08593?fields=title,year,venue,citationCount,influentialCitationCount): About 2,790 citations, 213 influential.
- **Stiennon RLHF** (2020): Combined supervised fine-tuning, a reward model and PPO; people preferred its summaries to the human-written references.
  - syllabus · [Stanford CS336 (Spring 2025), Lecture 15 'Alignment - SFT/RLHF' slides](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf): Slide 'More details and background from Stiennon'. The DPO derivation is presented as optimizing the implied reward 'via the Stiennon objective'. Ziegler 2019 …
  - syllabus · [Princeton COS 597G (Fall 2022)](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/): Recommended reading for 'Training LMs with human feedback'.
  - community · [Sebastian Raschka, 'LLM Training: RLHF and Its Alternatives'](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives): Credits Stiennon et al. with introducing the 'popular RLHF three-step procedure' that InstructGPT later used. Also listed among 'Key papers for LMs' in the HF …
  - other · [InstructGPT and Constitutional AI papers](https://arxiv.org/abs/2203.02155): InstructGPT: 'Our methodology follows that of Ziegler et al. (2019) and Stiennon et al. (2020)'. Constitutional AI describes itself as 'an extension of RLHF …
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2009.01325?fields=title,year,venue,citationCount,influentialCitationCount): About 3,575 citations, more than Ziegler 2019.
- **LoRA** (2021): Trains low-rank updates to frozen weights that merge back for inference with no extra latency; now the default PEFT method.
  - syllabus · [Stanford CS224N (2026) PEFT lecture; Princeton COS 597G](https://web.stanford.edu/class/cs224n/): Suggested reading in CS224N; recommended reading in Princeton's PEFT lecture.
  - adoption · [Hugging Face TRL docs, 'LoRA Without Regret' (not re-fetched)](https://huggingface.co/docs/trl/main/en/lora_without_regret): Documents the Thinking Machines (2025) finding that LoRA can match full fine-tuning for SFT and RL when configured correctly.
  - community · [daily.dev key papers reading list](https://daily.dev/agentic-ai-hub/key-papers-reading-list.md/): 'Parameter-efficient fine-tuning; why you can adapt huge models on one GPU.'
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2106.09685?fields=title,year,venue,citationCount,influentialCitationCount): About 24,510 citations, 3,408 influential.
- **FLAN** (2021): Introduced instruction tuning: fine-tuning on many tasks phrased as instructions lets a model solve unseen tasks zero-shot.
  - survey · [Lambert, RLHF Book, ch. Instruction Tuning](https://rlhfbook.com/c/09-instruction-tuning): Cited with T0 and Natural Instructions as the works that unified the task framework for instruction tuning.
  - other · [InstructGPT paper](https://arxiv.org/abs/2203.02155): Uses GPT-3 fine-tuned on FLAN (Wei et al., 2021) and T0 as the instruction-tuning baselines that RLHF is compared against.
  - other · [ICLR 2022 oral listing](https://iclr.cc/virtual/2022/oral/6255): Accepted as an oral at ICLR 2022.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2109.01652?fields=title,year,venue,citationCount,influentialCitationCount): About 5,523 citations, 488 influential; concurrent T0 has about 2,089.
- **InstructGPT** (2022): Applied the SFT, reward model and PPO recipe to general instruction following; ChatGPT was trained the same way.
  - survey · [Lambert, RLHF Book, ch. Related Works](https://rlhfbook.com/c/02-related-works): Applied RLHF to instruction following. The ChatGPT announcement said it was trained with InstructGPT's methods, with some differences in data collection.
  - community · [Sebastian Raschka, 'LLM Training: RLHF and Its Alternatives'; daily.dev reading list](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives): Raschka: 'this training pipeline is based on OpenAI's InstructGPT paper' and is 'widely considered to be the approach behind ChatGPT'. daily.dev tags it …
  - syllabus · [Stanford CS224N (2026) post-training lecture; Princeton COS 597G](https://web.stanford.edu/class/cs224n/): CS224N lists OpenAI's 'Aligning language models to follow instructions'. It is the required reading of Princeton's human-feedback lecture.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2203.02155?fields=title,year,venue,citationCount,influentialCitationCount): About 25,112 citations, 2,509 influential.
- **Constitutional AI** (2022): Replaced human harmlessness labels with AI feedback guided by a written set of principles (RLAIF).
  - survey · [Lambert, RLHF Book, ch. Constitutional AI and AI Feedback](https://rlhfbook.com/c/13-cai): 'the earliest documented, large-scale use of synthetic data for RLHF training'; 'kickstarted the broader field of RLAIF'; 'RLAIF has become a default method …
  - community · [Raschka, 'RLHF and Its Alternatives'; daily.dev reading list](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives): Raschka covers it first among the alternatives to RLHF. It is also on the daily.dev key papers list.
  - syllabus · [Stanford CS336 (Spring 2025), Lecture 15 slides](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf): Includes a slide taken 'From Constitutional AI, Bai et al.' in the AI-feedback part of the alignment lecture.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2212.08073?fields=title,year,venue,citationCount,influentialCitationCount): About 3,866 citations, 264 influential.
- **DPO** (2023): Turned the RLHF objective into a classification loss on preference pairs, with no reward model and no RL loop.
  - survey · [Lambert, RLHF Book, ch. Direct Alignment Algorithms](https://rlhfbook.com/c/12-direct-alignment): The most prominent direct alignment algorithm, which 'catalyzed an entire academic movement of aligning language models'. Lists its use in Zephyr-beta, Tulu …
  - adoption · [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783): 'We also explored on-policy algorithms such as PPO, but found that DPO required less compute for large-scale models and performed better'.
  - syllabus · [Stanford CS224N (2026) post-training lecture; Stanford CS336 Lecture 15](https://web.stanford.edu/class/cs224n/): Suggested reading in CS224N. CS336 gives DPO its own section ('DPO - RLHF without tears?').
  - community · [NeurIPS 2023 paper awards](https://blog.neurips.cc/tag/awards/): One of two Outstanding Main Track Runner-Ups.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2305.18290?fields=title,year,venue,citationCount,influentialCitationCount): About 11,151 citations, 2,313 influential.
- **QLoRA** (2023): Trains LoRA adapters on a frozen 4-bit quantized model, so a 65B model fits on one 48GB GPU.
  - adoption · [Hugging Face blog, 'Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA' (not re-fetched)](https://huggingface.co/blog/4bit-transformers-bitsandbytes): Announced 4-bit loading and QLoRA training in the HF stack.
  - community · [daily.dev key papers reading list](https://daily.dev/agentic-ai-hub/key-papers-reading-list.md/): '4-bit fine-tuning of 65B models on a single GPU; democratized adaptation.'
  - other · [NeurIPS 2023 oral listing](https://neurips.cc/virtual/2023/oral/73855): Presented as an oral at NeurIPS 2023.
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2305.14314?fields=title,year,venue,citationCount,influentialCitationCount): About 6,031 citations, 702 influential.
- **GKD** (2023): Trains the student on its own generated sequences with token-level teacher feedback, fixing the train-inference mismatch.
  - survey · [Song & Zheng, A Survey of On-Policy Distillation for LLMs](https://arxiv.org/html/2604.00626v3): GKD, 'alongside the concurrent MiniLLM (Gu et al., 2024), brought on-policy distillation into the LLM era in mid-2023'.
  - community · [awesome-on-policy-distillation curated list](https://github.com/chrisliu298/awesome-on-policy-distillation): In Foundations and in the 'Start Here' path: 'Unifying formulation spanning on-/off-policy mixtures with flexible divergences.'
  - adoption · [Gemma 2 technical report; Thinking Machines 'On-Policy Distillation' (Oct 2025)](https://arxiv.org/abs/2408.00118): Gemma 2: 'We also run distillation from the teacher on the student's distribution (Agarwal et al., 2024; Gu et al., 2024).' Thinking Machines: 'We extend prior …
  - citations · [Semantic Scholar API (not re-fetched)](https://api.semanticscholar.org/graph/v1/paper/arXiv:2306.13649?fields=title,year,venue,citationCount,influentialCitationCount): About 1,139 citations, 186 influential.

**Arrows** (the relation sentence shown on the site, then its source and quote):

- Christiano RLHF → Ziegler RLHF: Ziegler et al. took Christiano et al.'s preference reward model from simulators to text. ([source](https://arxiv.org/abs/1909.08593): “this work has only recently been applied to modern deep learning, and even then has only been applied to relatively simple simulated environments (Christiano …”)
- Ziegler RLHF → Stiennon RLHF: Stiennon et al. scaled up Ziegler's RLHF on summaries: bigger models, better human labels. ([source](https://arxiv.org/abs/2009.01325): “Compared to [73], we use significantly larger models, move to the batch setting for collecting human feedback, ensure high labeler-researcher agreement, and …”)
- Stiennon RLHF → InstructGPT: InstructGPT applied Stiennon's SFT, reward model and PPO steps to general instructions. ([source](https://arxiv.org/abs/2203.02155): “Our methodology follows that of Ziegler et al. (2019) and Stiennon et al. (2020), who applied it in the stylistic continuation and summarization domains.”)
- InstructGPT → DPO: DPO swapped InstructGPT's reward model and PPO steps for one simple classification loss. ([source](https://arxiv.org/abs/2305.18290): “We review the RLHF pipeline in Ziegler et al. (and later [40, 1, 28]). It usually includes three phases: 1) supervised fine-tuning (SFT); 2) preference …”)
- Adapters → LoRA: LoRA froze the model like Adapters, but its updates merge into the weights: no slowdown. ([source](https://arxiv.org/abs/2106.09685): “LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher …”)
- LoRA → QLoRA: QLoRA trained LoRA adapters on a frozen 4-bit model, shrinking fine-tuning memory. ([source](https://arxiv.org/abs/2305.14314): “QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA).”)
- Distillation → GKD: GKD kept Distillation's student-learns-from-teacher idea, but on text the student wrote. ([source](https://arxiv.org/html/2306.13649): “One of the prevalent techniques for compressing models is knowledge distillation (Hinton et al., 2015). ... GKD trains the student on its self-generated …”)

**Research core papers not on the roadmap:** Self-Instruct (med), MiniLLM (med), OPSD (med).

**Collection members not on the roadmap** (research-stage reasons):

- `opsd`: A research core paper (OPSD) that is not on the roadmap; see the curation notes.
- `sdpo`: A solid ICML 2026 poster (and, according to an author's page, an ICLR 2026 workshop best paper), but one of three concurrent January 2026 self-distillation papers. It has fewer citations than OPSD (Semantic Scholar: 332/73, not re-fetched, vs 462/113 for OPSD, re-fetched) and is listed after OPSD in the curated list. Its RLVR credit-assignment focus, benchmarked against GRPO, overlaps the RL roadmap. OPSD, a research core paper that is not on the roadmap, is the paper the research chose for the self-distillation branch. Listed as borderline.
- `distillm-2`: An incremental refinement of DistiLLM that adds a contrastive loss raising teacher-response likelihood and lowering student-response likelihood (ICML 2025 spotlight, re-fetched). Semantic Scholar lists about 97 citations (8 influential; not re-fetched). There is no evidence that it was adopted as a default recipe or that surveys or curated lists treat it as a turning point.
- `es-at-scale`: Added to the collection after the roadmap research; not assessed for the roadmap.
- `visual-domain-prompt`: Added to the collection after the roadmap research; not assessed for the roadmap.
- `neural-thickets`: Added to the collection after the roadmap research; not assessed for the roadmap.

**Borderline candidates** (research-stage reasons):

- HH-RLHF (Training a Helpful and Harmless Assistant) (2022, arXiv 2204.05862): First paper to add to this roadmap. Required reading in Princeton COS 597G, in the HF 'Illustrating RLHF' key papers, and the RLHF Book says Anthropic used it extensively for early Claude. Constitutional AI trains its preference model 'following the process in [Bai et al., 2022]', and DPO evaluates on the Anthropic HH data. Left out because it is concurrent with InstructGPT and uses essentially the same recipe; Stiennon 2020 is the shared ancestor of InstructGPT and Constitutional AI (only Stiennon → InstructGPT is drawn).
- Scaling Instruction-Finetuned Language Models (Flan-T5/Flan-PaLM) (2022, arXiv 2210.11416): CS224N 2026 and Princeton COS 597G both assign this paper rather than FLAN 2021, and Flan-T5 checkpoints were widely used (about 4,522 citations). It scales FLAN up rather than opening the line.
- T0 (Multitask Prompted Training) (2021, arXiv 2110.08207): Concurrent with FLAN and required reading in Princeton's instruction-tuning lecture, but it has fewer citations (about 2,089 vs 5,523) and a smaller downstream footprint.
- Sequence-Level Knowledge Distillation (2016, arXiv 1606.07947): Origin of training students on teacher-generated sequences, which is what SFT-on-teacher-output distillation (Alpaca, DeepSeek-R1-Distill) does. GKD and MiniLLM both use it as a baseline. Left out to keep the distillation line at Hinton, then on-policy KD.
- SDPO (Reinforcement Learning via Self-Distillation) (2026, arXiv 2601.20802): Concurrent with OPSD (ICML 2026 poster). According to an author's page it won a best-paper award at an ICLR 2026 workshop. It uses the feedback-conditioned model as a self-teacher for dense credit assignment and reports beating GRPO. It sits on the RL roadmap boundary and has fewer citations than OPSD per Semantic Scholar (332 vs 462; not re-fetched).
- SDFT (Self-Distillation Enables Continual Learning) (2026, arXiv 2601.19897): Concurrent self-distillation paper (ICML 2026 poster; ICLR 2026 workshop best paper per an author's page). The curated OPD list pairs it with OPSD as the 'no teacher at all' option. Could stand in for or join OPSD (not on the roadmap) as the self-distillation candidate if the 2026 wave settles differently.
- Thinking Machines, On-Policy Distillation (blog, Oct 2025) (2025): Popularized OPD as a cheap alternative to RL. OPSD cites it (Lu & Lab, 2025) more often than MiniLLM and takes its sampled-token reverse-KL variant from it. The curated list calls it the best single introduction. It is a blog post, not a paper, and its method extends GKD and MiniLLM.
- Prefix-Tuning / Prompt Tuning (2021, arXiv 2101.00190): Major early PEFT alternatives and the required readings of Princeton's PEFT lecture (LoRA was only recommended in 2022), with about 6.5k and 6.2k citations. They did not become the default for LLM post-training; LoRA did.
- SimPO (2024, arXiv 2405.14734): Covered in CS336's DPO-variants slide and the most cited DPO variant checked (about 1,224 citations, NeurIPS 2024). The RLHF Book calls it 'a minor change to the DPO optimization' and says algorithm choice matters far less than data and the initial model.
- Zephyr (2023, arXiv 2310.16944): The RLHF Book credits Zephyr-beta with starting DPO adoption in October 2023, and CS336 discusses it. It applies DPO and distillation rather than introducing a new method.
- LIMA (2023, arXiv 2305.11206): Influential 'superficial alignment hypothesis' (1,000 curated SFT examples; S2: 1,513 citations, 138 influential, NeurIPS 2023). It is a data-quality analysis rather than a method later work builds on.
- Tulu 3 (2024, arXiv 2411.15124): Fully open SFT, then DPO, then RLVR recipe that coined RLVR; discussed in CS336. Important as a recipe, but its new contribution (RLVR) belongs to the RL roadmap.

**Rejected notable candidates** (research-stage reasons):

- PPO: The optimizer inside Ziegler, Stiennon and InstructGPT, but a general RL algorithm that belongs on the reinforcement-learning roadmap (catalog: ppo). It fits as a cross-roadmap link into Ziegler 2019.
- DeepSeekMath (GRPO) / DeepSeek-R1 / DAPO / Dr. GRPO / GSPO: RL with verifiable rewards for reasoning; they belong to the RL collection, whose roadmap shows GRPO, DeepSeek-R1, DAPO and Dr. GRPO (GSPO is a research core paper that is not on it).
- Llama 2 / Llama 3 technical reports: Useful adoption evidence (Llama 3 chose DPO over PPO) but system reports, not methods later papers build on.
- KTO, IPO, ORPO, CPO and other DPO variants: Incremental changes to the DPO loss; the RLHF Book says algorithm choice matters far less than data and the starting model.
- SLiC-HF: Technically the first modern direct alignment algorithm, but per the RLHF Book it did not catch on. DPO is the paper the field built on.
- RLAIF (Lee et al., 2023): Scaled-up comparison of AI and human feedback that extends Constitutional AI rather than opening a new line.
- Stanford Alpaca / Vicuna: Influential model releases (blog posts, not papers) that apply Self-Instruct-style synthetic data and black-box distillation. Alpaca counts as adoption evidence for Self-Instruct, a research core paper that is not on the roadmap.
- DistilBERT: Applies Hinton-style KD to compress BERT during pretraining; not LLM post-training.
- WebGPT, Sparrow, GopherCite: RLHF applications to specific products or tasks; on RLHF reading lists but not built on as methods.
- Scaling Laws for Reward Model Overoptimization (Gao et al.): Important analysis of reward hacking, but a diagnostic study rather than a method on the lineage.
- AlpacaFarm; How Far Can Camels Go (Tulu 1): CS224N readings, but they are evaluation and simulation infrastructure or empirical studies of open instruction data, not post-training methods.
- A General Language Assistant as a Laboratory for Alignment (Askell et al., 2021): Princeton reading that framed the helpful-honest-harmless agenda and preference-model pretraining. It is a framing and analysis paper that HH-RLHF and Constitutional AI build on, and it is not itself the method line.
- DistiLLM (Ko et al., 2024): Skew-KL refinement of on-policy KD; listed among early OPD methods but not a turning point.

**Curation notes:** Selection rule: high-confidence papers first, then papers that start or connect a line. The research kept 15 papers. Ten are high; of the five medium papers, Stiennon et al. (connects the RLHF line) and Adapters (starts the PEFT line) were kept, and OPSD, Self-Instruct and MiniLLM are not on the roadmap (listed above). Ziegler → DPO is not drawn, because the path through Stiennon and InstructGPT already connects them. Stiennon → Constitutional AI is not drawn, because no layout without crossing or pass-through arrows exists with it. FLAN and Constitutional AI are shown without arrows.

## JEPA / World Models

Prediction in representation space: the latent world-model roots (World Models, PlaNet), the non-contrastive methods JEPAs rely on (BYOL, VICReg), image and video JEPAs, and JEPA-style world models that plan in latent space (DINO-WM, PLDM, V-JEPA 2, LeWorldModel).

| Paper | Year | Venue | Exercise | Confidence |
| --- | --- | --- | --- | --- |
| [World Models (NeurIPS 2018 version: Recurrent World Models Facilitate Policy Evolution, arXiv 1809.01999)](https://arxiv.org/abs/1803.10122) | 2018 | arXiv | grey | high |
| [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) | 2018 | ICML 2019 | grey | med |
| [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733) | 2020 | NeurIPS 2020 | grey | high |
| [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906) | 2021 | ICLR 2022 | `vicreg` | high |
| [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) | 2023 | CVPR 2023 | `i-jepa` | high |
| [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) | 2024 | TMLR 2024 | grey | high |
| [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://arxiv.org/abs/2411.04983) | 2024 | ICML 2025 | grey | high |
| [Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models](https://arxiv.org/abs/2502.14819) | 2025 | NeurIPS 2025 | grey | med |
| [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985) | 2025 | arXiv | grey | high |
| [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544) | 2025 | arXiv | `lejepa` | med |
| [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312) | 2026 | arXiv | `lewm` | med |

**Evidence per paper** (type, linked source, shortened note):

- **World Models** (2018): Learned a compact latent model of the environment from pixels and trained an agent inside its imagined rollouts.
  - survey · [Ding et al., Understanding World or Predicting Future? A Comprehensive Survey of World Models (ACM Computing Surveys)](https://arxiv.org/html/2411.14499v4): States that Ha et al. 'significantly revived and popularized the term "world model" in 2018'.
  - syllabus · [Essentials of Embodied AI Research - World Models key papers](https://essentials-of-eai-research.readthedocs.io/en/latest/world_models/key_papers/): Listed first among 28 papers and marked essential.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1803.10122?fields=title,year,venue,citationCount,influentialCitationCount): 2,238 citations (103 influential) for the arXiv record, plus 1,542 for the separately indexed NeurIPS version.
  - other · [LeWorldModel paper, Related Work](https://arxiv.org/html/2603.19312): 'World Models [26] pioneered learning policies directly from compact latent representations of high-dimensional observations.'
  - survey · [Survey companion repo tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model): First entry (RWM, NeurIPS 2018) in its roadmap of world models in the deep learning era.
- **PlaNet** (2018): Planned actions purely in a latent space learned from pixels, using model-predictive control instead of a policy network.
  - syllabus · [Essentials of Embodied AI Research - World Models key papers](https://essentials-of-eai-research.readthedocs.io/en/latest/world_models/key_papers/): Marked essential.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1811.04551?fields=title,year,venue,citationCount,influentialCitationCount): 2,270 citations, 268 influential.
  - other · [DINO-WM paper](https://arxiv.org/html/2411.04983v2): Cites Hafner et al. 2019 for world models trained on raw sensory data and among prior planning work. DINO-WM itself plans with CEM inside MPC but does not …
  - survey · [Survey companion repo tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model): Listed in the model-based RL lineage (PlaNet, ICML 2019), right after RWM.
- **BYOL** (2020): Avoided representation collapse without negative pairs, using a predictor and a slowly updated (EMA) target network.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2006.07733?fields=title,year,venue,citationCount,influentialCitationCount): 9,252 citations, 1,355 influential.
  - other · [V-JEPA paper](https://arxiv.org/html/2404.08471v1): Credits 'the non-contrastive collapse prevention strategy in BYOL (Grill et al., 2020)' as the strategy later made to work for image feature prediction, and …
  - community · [Yann LeCun interview, ICCV Daily 2023](https://www.rsipvision.com/ICCV2023-Thursday/): LeCun names BYOL first among the non-contrastive methods, followed by Barlow Twins, VICReg and I-JEPA.
  - other · [Dawid & LeCun, Introduction to Latent Variable Energy-Based Models: A Path Towards Autonomous Machine Intelligence (lecture notes, 2023)](https://arxiv.org/abs/2306.02572): Lists BYOL, SimSiam and Barlow Twins as approaches to regularizing JEPAs and related architectures.
- **VICReg** (2021): Made anti-collapse an explicit loss with variance, invariance and covariance terms, without EMA targets or negatives.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2105.04906?fields=title,year,venue,citationCount,influentialCitationCount): 1,564 citations, 259 influential.
  - other · [Dawid & LeCun lecture notes on the JEPA proposal (arXiv 2306.02572)](https://arxiv.org/abs/2306.02572): In the JEPA training section: 'let us give an example of how to maximize the information content of learned representations... The method is called …
  - adoption · [PLDM paper](https://arxiv.org/html/2502.14819v4): 'To prevent representation collapse, we use a VICReg-inspired objective, and inverse dynamics modeling.'
  - adoption · [LeJEPA paper](https://arxiv.org/html/2511.08544): 'LeJEPA recovers VICReg for this degenerate statistical test.'
  - community · [AbdelStark/awesome-jepa](https://github.com/AbdelStark/awesome-jepa): Listed in the JEPA list's Theory, Analysis and Recipes section as the anti-collapse regularizer.
  - community · [Yann LeCun interview, ICCV Daily 2023](https://www.rsipvision.com/ICCV2023-Thursday/): 'Then, we came up with two other ones called VICReg and I-JEPA.'
- **I-JEPA** (2023): Predicts representations of masked image blocks from a visible context block, without pixel reconstruction or hand-made augmentations.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2301.08243?fields=title,year,venue,citationCount,influentialCitationCount): 1,446 citations, 177 influential.
  - community · [Turing Post, 14 JEPA Milestones](https://www.turingpost.com/p/jepamap): Milestone 2, the first concrete model in the list.
  - adoption · [Hugging Face Transformers model docs](https://huggingface.co/docs/transformers/model_doc/ijepa): Built-in Transformers model with Meta checkpoints.
  - community · [AbdelStark/awesome-jepa](https://github.com/AbdelStark/awesome-jepa): First entry under Core Architectures.
- **V-JEPA** (2024): Extended JEPA to video: feature prediction alone learns representations of both motion and appearance.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2404.08471?fields=title,year,venue,citationCount,influentialCitationCount): 633 citations, 86 influential.
  - survey · [Ding et al., ACM Computing Surveys world-model survey](https://arxiv.org/html/2411.14499v4): 'recent implementations such as V-JEPA and V-JEPA2 have demonstrated the practical viability of video-based self-supervised learning'.
  - community · [Turing Post, 14 JEPA Milestones](https://www.turingpost.com/p/jepamap): Milestone 4: spatio-temporal masked feature prediction.
  - survey · [Survey companion repo tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model): Listed in the self-supervised world model section.
- **DINO-WM** (2024): Learned dynamics on frozen DINOv2 features from offline data and solved tasks zero-shot by planning in feature space.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2411.04983?fields=title,year,venue,citationCount,influentialCitationCount): 505 citations, 86 influential.
  - adoption · [Terver et al., What Drives Success in Physical Planning with Joint-Embedding Predictive World Models? (TMLR)](https://arxiv.org/abs/2512.24497): Abstract names DINO-WM and V-JEPA-2-AC as 'two established baselines'.
  - adoption · [Hierarchical Planning with Latent World Models (Zhang et al., 2026)](https://arxiv.org/html/2604.03208v1): Instantiates its planner on top of DINO-WM (Push-T), PLDM (mazes) and VJEPA2-AC (Franka).
  - adoption · [LeWorldModel paper](https://arxiv.org/html/2603.19312): Uses DINO-WM's PushT setup and dataset and its planning-solver configuration, and puts DINO-WM forward as the foundation-based family.
  - survey · [Survey companion repo tsinghua-fib-lab/World-Model](https://github.com/tsinghua-fib-lab/World-Model): Listed among self-supervised world models alongside V-JEPA and V-JEPA 2.
- **PLDM** (2025): Trained a JEPA world model end to end from pixels on reward-free data with a VICReg-style loss, then planned with it.
  - adoption · [LeWorldModel paper](https://arxiv.org/html/2603.19312): 'Among these baselines, PLDM is the closest to our setup, as it also learns a world model end-to-end directly from pixel observations.' LeWM's TwoRoom task is …
  - adoption · [Delta-JEPA (Zhang et al., Jun 2026; independent group)](https://arxiv.org/html/2606.31232v1): Uses PLDM, LeWM and Sub-JEPA as its Table 1 baselines.
  - adoption · [Hierarchical Planning with Latent World Models (Meta, 2026)](https://arxiv.org/html/2604.03208v1): Applies its hierarchical planner on top of PLDM for diverse-maze navigation.
  - adoption · [No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models (Boylan & Hokamp, 2026)](https://arxiv.org/html/2608.17542): Reports PLDM and DINO-WM as baselines: 'PLDM shows that a JEPA-style latent dynamics model can plan from reward-free offline data.'
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2502.14819?fields=title,year,venue,citationCount,influentialCitationCount): 118 citations, 23 influential; indexed as NeurIPS.
- **V-JEPA 2** (2025): Scaled video JEPA to a million hours of video, then added an action-conditioned predictor for zero-shot robot planning.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2506.09985?fields=title,year,venue,citationCount,influentialCitationCount): 922 citations, 131 influential in about 16 months.
  - adoption · [Meta AI V-JEPA 2 release blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/): Open release of code, checkpoints and physical-reasoning benchmarks.
  - adoption · [Terver et al., What Drives Success in Physical Planning with JEPA World Models? (TMLR)](https://arxiv.org/abs/2512.24497): Names V-JEPA-2-AC and DINO-WM as the two established baselines.
  - community · [Turing Post, 14 JEPA Milestones](https://www.turingpost.com/p/jepamap): Milestone 9: world model for understanding, prediction and planning.
- **LeJEPA** (2025): Argued isotropic Gaussian embeddings are optimal and enforced them with one regularizer, SIGReg, dropping training heuristics.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2511.08544?fields=title,year,venue,citationCount,influentialCitationCount): 284 citations, 65 influential within 11 months.
  - adoption · [LeWorldModel paper](https://arxiv.org/html/2603.19312): 'we adopt the Sketched-Isotropic-Gaussian Regularizer (SIGReg)' as its only anti-collapse term.
  - adoption · [No Gaussian Required (Boylan & Hokamp, 2026; independent group)](https://arxiv.org/html/2608.17542): Treats LeJEPA's SIGReg as the reference anti-collapse term and reproduces LeWM as its matched SIGReg control.
  - adoption · [LeVJEPA (Kuhn et al., Aug 2026; shares authors with LeJEPA)](https://arxiv.org/abs/2608.27395): 'the first video encoder trained under LeJEPA's collapse-free objective'; reports matching or beating V-JEPA 2 at 5.6-20.8x less pretraining compute. …
  - community · [Turing Post LeJEPA explainer and 14 JEPA Milestones](https://www.turingpost.com/p/jepamap): Milestone 10, for simplifying training with SIGReg.
- **LeWorldModel** (2026): Applied SIGReg to an action-conditioned JEPA: a world model trained end to end from pixels with two loss terms.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2603.19312?fields=title,year,venue,citationCount,influentialCitationCount): 304 citations, 88 influential within 7 months.
  - adoption · [Delta-JEPA (Zhang et al., Jun 2026; independent group)](https://arxiv.org/html/2606.31232v1): Calls LeWM 'Our primary baseline and foundation'.
  - adoption · [Fast LeWorldModel (Gao & Xu, 2026)](https://arxiv.org/abs/2606.26217): Direct extension that keeps LeWM's reconstruction-free JEPA setting and replaces step-by-step rollout with action-prefix prediction.
  - adoption · [Independent reproduction of LeWorldModel on TwoRoom (Singh, Aug 2026)](https://arxiv.org/abs/2608.10145): Reproduces the result (94.0% vs about 87% reported) but finds that success swings from 84% to 8% with the goal-construction protocol.
  - community · [Turing Post, 14 JEPA Milestones](https://www.turingpost.com/p/jepamap): Milestone 13: end-to-end world model from raw pixels.

**Arrows** (the relation sentence shown on the site, then its source and quote):

- World Models → PlaNet: PlaNet planned in a World Models-style latent model instead of training a policy in it. ([source](https://arxiv.org/abs/1811.04551): “a purely model-based agent that learns the environment dynamics from images and chooses actions through fast online planning in latent space ... We use the …”)
- PlaNet → DINO-WM: DINO-WM planned like PlaNet but in frozen DINOv2 features, without reconstructing pixels. ([source](https://arxiv.org/abs/2411.04983): “latent-space prediction is typically tied to objectives of reconstructing images (Hafner et al., 2019; Micheli et al., 2023; Hafner et al., 2024) ... DINO-WM …”)
- BYOL → I-JEPA: I-JEPA kept BYOL's slow-moving (EMA) target encoder but predicted masked block features. ([source](https://arxiv.org/abs/2301.08243): “The target-encoder weights are identical to the context-encoder weights at initialization, and updated via an exponential moving average thereafter [4, 18, 23 …”)
- I-JEPA → V-JEPA: V-JEPA carried I-JEPA's masked feature prediction loss from images over to video clips. ([source](https://arxiv.org/abs/2404.08471): “the objective in equation (1) is similar to the loss of Assran et al. (2023) used for image pretraining, but we modify it to use an ℓ1 regression, which we …”)
- V-JEPA → V-JEPA 2: V-JEPA 2 scaled up V-JEPA's pretraining and added an action-conditioned world model. ([source](https://arxiv.org/abs/2506.09985): “Section 2 describes the V-JEPA 2 pretraining procedure, including the key ingredients enabling scaling beyond the original V-JEPA recipe of Bardes et al. …”)
- VICReg → PLDM: PLDM kept its world model from collapsing with a VICReg-based loss extended over time. ([source](https://arxiv.org/abs/2502.14819): “To prevent collapse, we introduce a VICReg-based [4] objective. We modify it to apply variance objective across the time dimension to encourage features to …”)
- LeJEPA → LeWorldModel: LeWorldModel borrowed LeJEPA's rule that keeps a model from seeing all images as the same. ([source](https://arxiv.org/abs/2603.19312): “a trivial solution in which the encoder maps all inputs to a constant representation. To prevent this behavior, we introduce an anti-collapse regularization …”)
- PLDM → LeWorldModel: LeWorldModel trained end to end from pixels like PLDM, with two loss terms, not seven. ([source](https://arxiv.org/abs/2603.19312): “PLDM is the closest to our setup, as it also learns a world model end-to-end directly from pixel observations. However, it relies on a seven-term training …”)
- VICReg → LeJEPA: LeJEPA widened VICReg's rule for keeping features varied, shaping them into a bell curve. ([source](https://arxiv.org/html/2511.08544): “Lastly, it is possible to recover some existing SSL frameworks in the limit by employing LeJEPA with a particular test–instead of the preferred (Epps–Pulley). …”)

**Research core papers not on the roadmap:** A Path Towards AMI (JEPA position paper) (high).

**Collection members not on the roadmap** (research-stage reasons):

- `visreg`: June 2026 preprint with no venue, co-authored by Balestriero. It replaces VICReg's covariance term with a sliced-Wasserstein sketch, which refines VICReg/SIGReg. 16 citations (4 influential), and no independent discussion or adoption found. Too early and too incremental for the core roadmap.
- `causal-jepa`: ICML 2026 (confirmed on arXiv). An object-level-masking extension of JEPA world models from the LeWM group that specializes the line rather than opening it. 20 Semantic Scholar citations (3 influential) eight months after release, and a Turing Post milestone listing, but no independent adoption found. It can stay in the collection as a recent extension hanging off LeWM, but not as a core node.

**Borderline candidates** (research-stage reasons):

- Barlow Twins (2021, arXiv 2103.03230): 3,329 citations (486 influential), ICML 2021, and the direct precursor of VICReg ('VICReg uses the same decorrelation mechanism as Barlow Twins'). No JEPA paper reviewed (I-JEPA, V-JEPA, PLDM, LeJEPA, LeWM) builds on it directly, and the JEPA-specific lists (awesome-jepa, Turing Post's 14 milestones) omit it. VICReg represents the explicit-regularizer family here; Barlow Twins belongs in a general SSL roadmap.
- data2vec (2022, arXiv 2202.03555): 1,280 citations (222 influential); I-JEPA says 'Closest to our work is data2vec and Context Autoencoders' (masked latent prediction with an EMA teacher). A strong pre-JEPA ancestor, but later JEPA work builds on I-JEPA, not data2vec, and BYOL already represents the EMA-target idea.
- Dreamer (Dream to Control) (2019, arXiv 1912.01603): 2,450 citations (302 influential), marked essential in the EAI reading list, and cited by LeWM as the task-specific family. It is the reconstruction- and reward-driven model-based RL line that JEPA world models position against; PlaNet represents that root here.
- DreamerV3 (2023, arXiv 2301.04104): 1,517 citations; singled out by the ACM CSUR survey (over 150 tasks) and DINO-WM's main comparison point. A refinement inside the Dreamer line that fits better in a model-based RL roadmap.
- TD-MPC (2022, arXiv 2203.04955): 545 citations (71 influential). A decoder-free latent world model with MPC. LeWM cites TD-MPC/TD-MPC2 (not PlaNet) for latent-space MPC, and PLDM cites it for reconstruction-free control. It stays out because its latent is shaped by reward/value, and LeWM classes it as task-specific. It is the strongest alternative to PlaNet as the planning root.
- DINO (2021, arXiv 2104.14294): 10,639 citations. I-JEPA cites it for the EMA target encoder being essential with ViTs, and LeJEPA reuses its multi-view setup. Left out because it would anchor a general SSL roadmap and BYOL represents the EMA-target idea here.
- DINOv2 (2023, arXiv 2304.07193): 11,376 citations; the frozen encoder underlying DINO-WM. A dependency of the line, not part of the JEPA lineage.
- SimSiam (2020, arXiv 2011.10566): 5,332 citations; showed stop-gradient is the key anti-collapse ingredient. A simplification of BYOL, which is kept.
- CPC (Contrastive Predictive Coding) (2018, arXiv 1807.03748): 14,880 citations; predicts future latents and appears in JEPA lineage write-ups, but it is contrastive (InfoNCE), which JEPA moves away from.
- C-SWM (Contrastively-Trained Structured World Models) (2019, arXiv 1911.12247): 365 citations; an early reconstruction-free latent world model listed in the EAI reading list. Contrastive, and none of the JEPA world-model papers reviewed name it as a basis.
- V-JEPA 2.1 (2026, arXiv 2603.14482): 125 citations (45 influential) in about 7 months and a Turing Post milestone, but an incremental upgrade (dense features) to V-JEPA 2.
- Image World Models (IWM) (2024, arXiv 2403.00504): Turing Post-adjacent coverage and listed in awesome-jepa, but only 83 citations (5 influential) and little follow-up use.
- Intuitive physics emerges from V-JEPA pretraining (2025, arXiv 2502.11831): A widely discussed analysis (86 citations, 4 influential) showing V-JEPA flags physically implausible events. An evaluation finding rather than a method others build on.
- SPR (Self-Predictive Representations) (2020, arXiv 2007.05929): 461 citations; BYOL-style latent future prediction for RL, a JEPA-like idea before the term existed. It sits in the sample-efficient RL line, and no JEPA paper reviewed names it as a basis.

**Rejected notable candidates** (research-stage reasons):

- MC-JEPA (2023): Early motion-and-content video JEPA from the VICReg authors. Superseded by V-JEPA, with little follow-up use.
- JEPAs Focus on Slow Features (Sobal et al., 2022): Early small-scale study of VICReg-trained JEPA world models. Its line continues in PLDM, which is the node kept.
- Modality ports: A-JEPA, Audio-JEPA, Point-JEPA, 3D-JEPA, ACT-JEPA, ECG/LiDAR/graph/atomistic JEPAs: Applications of I-JEPA/V-JEPA to new data types; awesome-jepa's domain sections are mostly this kind of incremental port.
- LLM-JEPA (2025) and VL-JEPA (2025): Early attempts to bring JEPA to language and vision-language, with no evidence of adoption yet.
- MAE (Masked Autoencoders, 2021): The pixel-reconstruction counterpart that I-JEPA defines itself against. Belongs to a masked-image-modeling roadmap.
- SimCLR / MoCo: Contrastive methods; both are members of the Visual Generation collection (see its off-roadmap list). JEPA's lineage moves away from negatives.
- Generative world models: Genie, Cosmos, Navigation World Models, DIAMOND, Dreamer 4, video diffusion / Sora: They predict pixels or tokens rather than embeddings, which puts them out of this roadmap's latent-prediction scope.
- MuZero (2019): A landmark learned model for planning, but its latent is trained by reward/value/policy targets. No JEPA paper reviewed names it as a basis.
- 2026 JEPA world-model extensions: Hierarchical Planning with Latent World Models (HWM), ThinkJEPA, Delta-JEPA, Fast-LeWM, Flow-JEPA, Var-JEPA, Gaussian Joint Embeddings, LeVJEPA: Too recent to show importance beyond author-adjacent coverage; most are refinements of DINO-WM/PLDM/LeWM or of LeJEPA.
- When Does LeJEPA Learn a World Model? (2026): Theory follow-up by the LeJEPA authors that supports LeJEPA's importance but is itself a refinement.
- What Drives Success in Physical Planning with JEPA World Models? (Terver et al., TMLR): Useful empirical study and code base (jepa-wms; 54 citations), but an analysis of existing JEPA world models rather than a new line.

**Curation notes:** Of the 12 researched papers, 11 are kept: Path to AMI (LeCun's position paper, OpenReview only) is left out because every roadmap paper must have an arXiv version. DINO-WM → LeWorldModel is not drawn, because LeWorldModel already has two incoming arrows (LeJEPA, PLDM). VICReg → LeJEPA is drawn because no other path on the map connects them. Causal-JEPA and VISReg stay in the collection list.

## Visual Generation

Image generation outside the diffusion-sampling line: the VAE and GAN foundations, pixel-level autoregression, discrete visual tokenizers (VQ-VAE, VQGAN), text-to-image with tokens (DALL-E) and the CLIP embedding generators rely on, masked, scale-wise and continuous-token generation (MaskGIT, VAR, MAR), and representation alignment (REPA).

| Paper | Year | Venue | Exercise | Confidence |
| --- | --- | --- | --- | --- |
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | 2013 | ICLR 2014 | `vae` | high |
| [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | 2014 | NeurIPS 2014 | `gan` | high |
| [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) | 2016 | ICML 2016 | grey | high |
| [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) | 2017 | NeurIPS 2017 | grey | high |
| [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) | 2018 | CVPR 2019 | grey | high |
| [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) | 2020 | CVPR 2021 | grey | high |
| [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | 2021 | ICML 2021 | `clip` | high |
| [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) | 2021 | ICML 2021 | grey | high |
| [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200) | 2022 | CVPR 2022 | grey | high |
| [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905) | 2024 | NeurIPS 2024 | `var` | high |
| [Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838) | 2024 | NeurIPS 2024 | grey | high |
| [Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/abs/2410.06940) | 2024 | ICLR 2025 | `repa` | high |

**Evidence per paper** (type, linked source, shortened note):

- **VAE** (2013): Made latent-variable models trainable by backpropagation; later image tokenizers build on its encoder-decoder design.
  - other · [ICLR 2024 Test of Time Award announcement](https://blog.iclr.cc/?p=751): 'Auto-Encoding Variational Bayes' (Kingma & Welling) is listed as the Test of Time winner, selected from ICLR 2013/2014 papers.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1312.6114?fields=title,year,citationCount,influentialCitationCount): 17,261 citations, 3,106 influential.
  - syllabus · [MIT 6.S978 Deep Generative Models (Kaiming He), Fall 2024](https://mit-6s978.github.io/schedule.html): Has a dedicated VAE lecture (09/12); CMU 10-423 also has a VAE lecture.
  - survey · [Bond-Taylor et al., Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models (TPAMI)](https://arxiv.org/abs/2103.04922): VAEs are one of the five model families the review is organized around.
- **GAN** (2014): Trained a generator against a discriminator; GANs led image synthesis until diffusion models took over.
  - other · [NeurIPS 2024 Test of Time Paper Awards](https://blog.neurips.cc/2024/11/27/announcing-the-neurips-2024-test-of-time-paper-awards/): Test of Time award; called 'one of the foundational pieces for generative modeling', cited more than 85,000 times.
  - survey · [Chakraborty et al., Ten Years of Generative Adversarial Nets (GANs): A Survey of the State-of-the-Art (Machine Learning: Science and Technology, 2024)](https://arxiv.org/abs/2308.16316): A decade-long survey organized around the GAN framework introduced in 2014 and its variants.
  - syllabus · [CMU 10-423/623 Generative AI, Spring 2025](https://www.cs.cmu.edu/~mgormley/courses/10423-s25/schedule.html): 'Generative Adversarial Nets' (Goodfellow et al., 2014) is the assigned reading for the GAN lecture; MIT 6.S978 also has a GAN lecture.
- **PixelRNN** (2016): Generated images pixel by pixel with exact likelihood; a root of later visual autoregressive models.
  - other · [Google Research blog, ICML 2016 research at Google](https://www.research.google/blog/icml-2016-research-at-google/): Marked as an ICML 2016 Best Paper Award recipient; the TWiML ICML 2016 recap …
  - survey · [Autoregressive Models in Vision: A Survey (arXiv 2411.05902)](https://arxiv.org/html/2411.05902v2): PixelRNN 'emerged as the pioneering work for AR image generation'; it anchors the survey's pixel-based category.
  - syllabus · [MIT 6.S978 Deep Generative Models](https://mit-6s978.github.io/schedule.html): Required reading in the 09/24 Autoregressive Models session.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1601.06759?fields=title,year,citationCount,influentialCitationCount): 2,932 citations, 255 influential; the follow-up Gated PixelCNN paper has a further 2,846.
- **VQ-VAE** (2017): Learned a discrete codebook of image tokens, then modeled the tokens: the tokenize-then-generate recipe.
  - survey · [Autoregressive Models in Vision: A Survey](https://arxiv.org/html/2411.05902v2): 'A seminal work on discrete representation is VQVAE'; VQ-VAE, VQ-VAE-2 and VQGAN 'collectively established a two-stage paradigm for next-token prediction.'
  - survey · [A Comprehensive Survey of Discrete Tokenizers (arXiv 2502.12448)](https://arxiv.org/html/2502.12448v1): VQ-VAE and VQ-GAN are described as 'the pioneers in image generation' among tokenizers.
  - community · [Sander Dieleman, 'Generative modelling in latent space' (blog, 2025)](https://sander.ai/2025/04/15/latents.html): 'Autoregressive models of images took a huge leap forward with the seminal VQ-VAE paper.'
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1711.00937?fields=title,year,citationCount,influentialCitationCount): 8,198 citations, 1,054 influential.
- **StyleGAN** (2018): Redesigned the GAN generator with style modulation, producing photorealistic faces and an editable latent space.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1812.04948?fields=title,year,citationCount,influentialCitationCount): 13,727 citations, 2,128 influential, more than BigGAN (6,329) or ProGAN (8,852); DCGAN is comparable (15,340 / 1,762).
  - adoption · [R3GAN, 'The GAN is dead; long live the GAN! A Modern GAN Baseline' (NeurIPS 2024)](https://arxiv.org/abs/2501.05441): 'Using StyleGAN2 as an example, we present a roadmap of simplification and modernization'; it is an MIT 6.S978 required reading, together with StyleGAN-T.
  - survey · [Ten Years of GANs survey](https://arxiv.org/abs/2308.16316): The abstract names StyleGAN among the major GAN variants it summarizes.
  - community · [awesome-image-synthesis (Victarry) GAN architecture lineage graph](https://github.com/Victarry/awesome-image-synthesis): Lineage chart VanillaGAN -> DCGAN -> PG-GAN -> StyleGAN -> StyleGAN2/StyleGAN3, with StyleGAN as the hub of later GAN architectures.
- **VQGAN** (2020): Added adversarial and perceptual losses to VQ-VAE for sharp tokens modeled by a transformer; the default tokenizer.
  - survey · [Autoregressive Models in Vision: A Survey](https://arxiv.org/html/2411.05902v2): 'A significant milestone that unlocked the potential of autoregressive visual generation was the advent of VQ-GAN.'
  - community · [Sander Dieleman, 'Generative modelling in latent space'](https://sander.ai/2025/04/15/latents.html): 'VQGAN became a core technology enabling the rapid progress in generative modelling of perceptual signals. Its impact cannot be overestimated.' Rombach et al. …
  - adoption · [LlamaGen, MaskGIT and VAR papers](https://arxiv.org/html/2406.06525v1): LlamaGen: 'We use the same architecture as VQGAN'. MaskGIT: 'the same setup for the first stage as in the VQGAN model'. VAR: 'the same architecture as VQGAN …
  - community · [VQGAN-CLIP (ECCV 2022)](https://arxiv.org/abs/2204.08583): The open text-guided generation method built on VQGAN ('using CLIP to guide VQGAN') that spread through the 2021 generative-art community.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2012.09841?fields=title,year,citationCount,influentialCitationCount): 4,718 citations, 707 influential.
- **CLIP** (2021): Learned a shared image-text embedding that generators use for text conditioning, reranking and evaluation.
  - adoption · [Stable Diffusion v1-4 model card](https://huggingface.co/CompVis/stable-diffusion-v1-4): 'uses a fixed, pretrained text encoder (CLIP ViT-L/14)'; 'Text prompts are encoded through a ViT-L/14 text-encoder.'
  - adoption · [DALL-E paper](https://ar5iv.labs.arxiv.org/html/2102.12092): 'we rerank the samples drawn from the transformer using a pretrained contrastive model (Radford et al. 2021).'
  - community · [VQGAN-CLIP paper (ECCV 2022)](https://arxiv.org/abs/2204.08583): 'Using CLIP to guide VQGAN' became the open text-to-image method of the 2021 generative-art community.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2103.00020?fields=title,year,citationCount,influentialCitationCount): 58,002 citations, 10,660 influential.
- **DALL-E** (2021): Scaled the two-stage recipe to open-domain text-to-image with a 12B transformer over text and image tokens.
  - survey · [Autoregressive Models in Vision: A Survey](https://arxiv.org/html/2411.05902v2): 'A pioneering effort in this field is exemplified by DALL·E' (text-to-image AR).
  - syllabus · [MIT 6.S978 'AR and Tokenizers' session](https://mit-6s978.github.io/schedule.html): Listed as an optional recommended reading ('Zero-Shot Text-to-Image Generation (DALLE1)').
  - other · [VAR and LlamaGen introductions](https://arxiv.org/html/2404.02905v2): VAR: 'Trailblazing efforts like VQGAN and DALL-E'. LlamaGen (https://arxiv.org/html/2406.06525v1) lists VQVAE, VQGAN and DALL-E as the 'pioneering works' of AR …
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2102.12092?fields=title,year,citationCount,influentialCitationCount): 6,810 citations, 403 influential.
- **MaskGIT** (2022): Predicts masked image tokens in parallel over a few refinement steps instead of one token at a time.
  - other · [eMIGM, Effective and Efficient Masked Image Generation Models (arXiv 2503.07197)](https://arxiv.org/pdf/2503.07197): 'In image generation, MaskGIT (Chang et al., 2022) introduced masked image generation.'
  - adoption · [Muse: Text-To-Image Generation via Masked Generative Transformers](https://ar5iv.labs.arxiv.org/html/2301.00704): Trains with a cosine masking rate 'as was done in (Chang et al. 2022)' and decodes with the same cosine schedule; Muse has 815 citations.
  - adoption · [MAR paper](https://arxiv.org/html/2406.11838v3): 'masked generative models, e.g., MaskGIT and MAGE, can be generalized under the broad concept of autoregression.'
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2202.04200?fields=title,year,citationCount,influentialCitationCount): 1,351 citations, 235 influential.
- **VAR** (2024): Generates images coarse to fine by next-scale prediction; a GPT-style model that beat diffusion transformers on ImageNet.
  - other · [NeurIPS 2024 Best Paper Awards](https://blog.neurips.cc/2024/12/10/announcing-the-neurips-2024-best-paper-awards/): Main-track best paper.
  - survey · [Autoregressive Models in Vision: A Survey](https://arxiv.org/html/2411.05902v2): 'A pioneering work that achieved this breakthrough is VAR'; it is the representative of the scale-based category.
  - syllabus · [MIT 6.S978 'AR and Tokenizers' session](https://mit-6s978.github.io/schedule.html): Required reading.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2404.02905?fields=title,year,citationCount,influentialCitationCount): 1,230 citations, 198 influential in about 2.5 years.
  - community · [NeurIPS 2025: 'Multi-scale Autoregressive Models are Laplacian, Discrete, and Latent Diffusion Models In Disguise'](https://neurips.cc/virtual/2025/125493): A follow-up analysis paper that revisits VAR, showing it became a reference point.
- **MAR** (2024): Showed autoregressive image models can generate continuous tokens with a small per-token diffusion head, no codebook.
  - syllabus · [MIT 6.S978 'AR and Diffusion' session](https://mit-6s978.github.io/schedule.html): Required reading.
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2406.11838?fields=title,year,citationCount,influentialCitationCount): 774 citations, 147 influential.
  - adoption · [Fluid (ICLR 2025), scaling continuous-token AR text-to-image](https://arxiv.org/html/2410.13863v1): 'we utilize the Diffusion Loss (Li et al., 2024)' and 'strictly followed the open-sourced code of Li et al. (2024)'.
  - survey · [Autoregressive Models in Vision: A Survey](https://arxiv.org/html/2411.05902v2): MAR 'provides a theoretical analysis suggesting that raster order and discrete representations are not necessary'.
- **REPA** (2024): Aligns a diffusion transformer's hidden states with frozen pretrained features, speeding up training over 17x.
  - other · [ICLR 2025 oral](https://iclr.cc/virtual/2025/oral/31896): Selected for an oral presentation (also stated in the arXiv comments).
  - community · [The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/pretrained-embeddings-accelerate-diffusion-transformers-learning): Covered as a notable advance in training diffusion transformers.
  - adoption · [REPA-E (ICCV 2025, arXiv 2504.10483)](https://arxiv.org/html/2504.10483v3): Uses the REPA loss to make joint VAE and diffusion training work; 249 citations (Semantic Scholar).
  - citations · [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2410.06940?fields=title,year,citationCount,influentialCitationCount): 796 citations, 170 influential in two years.

**Arrows** (the relation sentence shown on the site, then its source and quote):

- VAE → VQ-VAE: VQ-VAE kept the VAE's encoder-decoder but made the latent codes discrete via a codebook. ([source](https://arxiv.org/abs/1711.00937): “Our model, the Vector Quantised-Variational AutoEncoder (VQ-VAE), differs from VAEs in two key ways: the encoder network outputs discrete, rather than …”)
- PixelRNN → VQ-VAE: VQ-VAE used PixelCNN, from the PixelRNN paper, to generate its codes one by one. ([source](https://ar5iv.labs.arxiv.org/html/1711.00937): “PixelCNNs [29, 38] are convolutional autoregressive models ... We use a PixelCNN over the discrete latents for images”)
- GAN → StyleGAN: StyleGAN kept GAN training but rebuilt the generator with ideas from style transfer. ([source](https://arxiv.org/abs/1812.04948): “We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature.”)
- GAN → VQGAN: VQGAN trained its tokenizer against a GAN discriminator so decoded images stay sharp. ([source](https://ar5iv.labs.arxiv.org/html/2012.09841): “we propose VQGAN, a variant of the original VQVAE, and use a discriminator and perceptual loss [40, 30, 39, 17, 47] to keep good perceptual quality at …”)
- VQ-VAE → VQGAN: VQGAN kept VQ-VAE's codebook of image tokens but modeled the tokens with a transformer. ([source](https://ar5iv.labs.arxiv.org/html/2012.09841): “Our approach uses a convolutional VQGAN to learn a codebook of context-rich visual parts, whose composition is subsequently modeled with an autoregressive …”)
- VQ-VAE → DALL-E: DALL-E followed VQ-VAE's two stages: compress images to discrete tokens, then model them. ([source](https://ar5iv.labs.arxiv.org/html/2102.12092): “using a two-stage training procedure, similar to (Oord et al. 2017; Razavi et al. 2019): ... We train a discrete variational autoencoder (dVAE) ... train an …”)
- CLIP → DALL-E: DALL-E used CLIP to score its sampled images against the caption and keep the best. ([source](https://ar5iv.labs.arxiv.org/html/2102.12092): “we rerank the samples drawn from the transformer using a pretrained contrastive model (Radford et al. 2021). Given a caption and a candidate image, the …”)
- VQGAN → MaskGIT: MaskGIT kept VQGAN's tokenizer but filled in masked tokens in parallel, not one by one. ([source](https://ar5iv.labs.arxiv.org/html/2202.04200): “Since our goal is to improve the second stage, we employ the same setup for the first stage as in the VQGAN model ... MaskGIT learns to predict randomly masked …”)
- VQGAN → VAR: VAR gave VQGAN's tokenizer multi-scale quantization to generate images coarse to fine. ([source](https://ar5iv.labs.arxiv.org/html/2404.02905): “We employ the same architecture as VQGAN [30] but with a modified multi-scale quantization layer. ... redefines the autoregressive learning on images as …”)
- MaskGIT → MAR: MAR kept MaskGIT's parallel masked prediction but used continuous tokens, not a codebook. ([source](https://ar5iv.labs.arxiv.org/html/2406.11838): “Following MaskGIT [4], our MAR enjoys the flexibility of predicting multiple tokens at a time. ... This approach eliminates the need for discrete-valued …”)

**Research core papers not on the roadmap:** DINOv2 (med).

**Collection members not on the roadmap** (research-stage reasons):

- `simclr`: Contrastive self-supervised learning for recognition (26,740 citations), not a generation method. It is in this collection because Drifting Models computes its drifting field in a self-supervised feature space; the paper reports that public SimCLR and MoCo v2 encoders give 'decent results', but its latent-space MAE performs best. It fits as a prerequisite of the Drifting exercise or in a self-supervised representation collection (none exists besides JEPA / World Models).
- `moco`: Same situation as SimCLR. MoCo features appear only as alternative encoders: MoCo v2 in Drifting's ablations and MoCov3 in REPA's encoder comparison, where DINOv2 is the reference encoder. The generation field converged on DINOv2, so MoCo is a prerequisite at most, not a roadmap node.
- `rotation-trick`: ICLR 2025 oral that improves gradient flow through vector quantization compared with the straight-through estimator (68 citations, 6 influential). It refines VQ-VAE/VQGAN training, and there is no evidence yet that it became a default. It fits as an exercise hanging off the VQ-VAE/VQGAN nodes (it is also a member of the Retrieval collection).
- `drifting`: Promising and heavily discussed: a NeurIPS 2026 Oral per its project page, with 129 citations (63 influential) and several follow-ups in about 8 months. It is still too new to show downstream adoption. It is the top borderline candidate, to be re-checked for promotion after NeurIPS 2026.
- `fd-loss`: April 2026 objective that makes a Fréchet distance in representation space trainable (0.72 FID one-step on ImageNet 256). It has 42 citations (11 influential), no listed venue, and no adoption evidence yet. It is also a member of the Diffusion & Flow collection, next to the one-step distillation methods.

**Borderline candidates** (research-stage reasons):

- Drifting Models (2026, arXiv 2602.04770): Strongest candidate for promotion. A non-adversarial one-step generator trained by a drifting field computed in a self-supervised feature space (ImageNet 256: 1.54 FID at 1 NFE). The project page lists it as a NeurIPS 2026 Oral. It has 129 citations, 63 of them influential, after about 8 months (Semantic Scholar), and follow-ups already include DriftXpress (2605.12183), policy-learning variants (2605.07727, 2604.04528) and theory papers recasting it as score matching (2603.07514). It was widely discussed on X (e.g., https://x.com/iScienceLuvr/status/2019340015029919925). It is too recent to show downstream adoption, and its own related work places it near MMD moment matching. A re-check is due after NeurIPS 2026 and in 2027.
- DCGAN (2015, arXiv 1511.06434): Set the convolutional GAN recipe (ICLR 2016). 15,340 citations, 1,762 influential; the 10-year GAN survey says it 'marks a significant breakthrough in the realm of generative AI, particularly for image generation', and the awesome-image-synthesis lineage graph puts it between the original GAN and PG-GAN/StyleGAN. Kept out only so the GAN branch has a single architecture node. StyleGAN builds directly on ProGAN, not DCGAN, so adding DCGAN cleanly would also need ProGAN.
- Latent Diffusion Models (LDM) (2021, arXiv 2112.10752): 28,275 citations, and its KL/VQ autoencoders are the most widely used visual tokenizers (MAR's KL-16 comes from the LDM codebase). Its main contribution, however, is diffusion in latent space, so it belongs on the Diffusion & Flow roadmap with a cross-roadmap edge from VQGAN.
- MAGVIT-v2 (LFQ) (2023, arXiv 2310.05737): The 'tokenizer is key' paper that introduced lookup-free quantization. It is an MIT 6.S978 required reading and has 727 citations; Dieleman notes that LFQ, FSQ and BSQ 'have gained traction'. It is the best candidate for a 'tokenizer refinements' node (which would anchor the rotation-trick exercise), but it refines VQGAN rather than opening a line.
- LlamaGen (2024, arXiv 2406.06525): 841 citations. Open models showing that a plain Llama-style AR transformer over VQGAN-architecture tokens beats LDM and DiT, and a widely used baseline, but it introduces no new mechanism beyond VQGAN plus GPT.
- RQ-VAE / RQ-Transformer (2022, arXiv 2203.01941): Residual quantization; 1,009 citations. VAR calls its residual design 'akin to' RQ-Transformer, so it is a plausible direct precursor of VAR's multi-scale tokenizer. The survey evidence treats it as an improvement on VQ, not a turning point.
- FSQ (2023, arXiv 2309.15505): Replaces VQ with finite scalar quantization (ICLR 2024); 639 citations; an MIT 6.S978 optional reading. A simplification of the VQ step rather than a new paradigm.
- TiTok (1D tokenizer) (2024, arXiv 2406.07550): Compresses an image to 32 one-dimensional tokens (NeurIPS 2024); an MIT 6.S978 required reading; 341 citations. It is too early to tell whether 1D tokenization becomes the default.
- Representation Autoencoders (RAE) (2025, arXiv 2510.11690): ICLR 2026 poster; 316 citations (61 influential) in a year. It replaces the VAE with a frozen pretrained encoder (DINO, SigLIP, MAE) plus a trained decoder, and explicitly avoids REPA-style 'auxiliary representation alignment losses'. It is the natural successor to DINOv2 and REPA if adoption continues.
- Unified token-based multimodal generators (Emu3; Chameleon) (2024, arXiv 2409.18869): Emu3 (821 citations; published in Nature in 2026) and Chameleon (1,015 citations) generate images by plain next-token prediction over VQ tokens inside one multimodal model, which extends this roadmap's token branch. PaperLeet has no multimodal collection, so whether this roadmap should carry one representative node is open.
- Parti (2022, arXiv 2206.10789): Scaled AR text-to-image to 20B parameters (TMLR); 1,587 citations. It is closed and was quickly overtaken; DALL-E already represents the AR text-to-image step.
- BigGAN (2018, arXiv 1809.11096): 6,329 citations; the class-conditional ImageNet GAN baseline still reported in VAR and MaskGIT tables. StyleGAN represents the GAN-architecture branch instead.
- FID (TTUR) (2017, arXiv 1706.08500): The standard evaluation metric for every visual generator and the starting point of FD-Loss. It is a metric rather than a generation method, so it is better shown as a shared prerequisite. The Semantic Scholar arXiv record (517 citations) is split from the conference version, so the count is not comparable.
- MAE (2021, arXiv 2111.06377): 13,225 citations. MAR adopts 'the bidirectional attention implementation as done in MAE', and a latent-space MAE is Drifting Models' best feature encoder. Its primary home is a representation-learning roadmap.

**Rejected notable candidates** (research-stage reasons):

- WGAN: A GAN training-stability refinement, superseded by later regularizers, with no lineage into current tokenizer or AR pipelines.
- ProGAN: Direct precursor of StyleGAN (8,852 citations), folded into the StyleGAN node.
- StyleGAN2 / StyleGAN3: Refinements of StyleGAN (StyleGAN2: 7,295 citations), credited inside the StyleGAN node (R3GAN baseline).
- VQ-VAE-2: Hierarchical refinement of VQ-VAE (2,518 citations); VQGAN is the refinement the field adopted as its default tokenizer.
- ViT-VQGAN: Transformer-backbone refinement of VQGAN (847 citations); Muse chose VQGAN over it, and it did not become the default tokenizer.
- Muse: Text-to-image scale-up of MaskGIT (815 citations) that reuses its masking schedule; counted as MaskGIT adoption evidence.
- Infinity / HART / other VAR follow-ups: Scale-ups and variants of next-scale prediction (Infinity: 340 citations); they refine VAR.
- GIVT / MAGE: GIVT (87 citations) was an earlier continuous-token AR model and MAGE (301) a masked generative encoder; MAR subsumed both lines and is the node the field cites.
- REPA-E / VA-VAE: Representation-alignment refinements (249 and 449 citations); credited to REPA and to DINOv2 (a research core paper that is not on the roadmap).
- Image Transformer / iGPT: Pixel-level transformer autoregression, superseded by token-based AR. iGPT (an MIT 6.S978 optional reading) matters more for representation learning.
- DiT / SiT: Diffusion backbones (DiT: 7,763 citations); DiT is assessed under Diffusion & Flow but is not on that roadmap, and both enter here only as the base models REPA regularizes.
- DALL-E 2 / Imagen: Diffusion-based text-to-image systems, assessed (and not included) under Diffusion & Flow.
- Transfusion / Show-o / Janus: Hybrid unified multimodal models (MIT 6.S978 readings, 510-810 citations) that mix diffusion and AR inside an LLM; better placed on a multimodal roadmap. Emu3/Chameleon is the borderline representative.
- pix2pix / CycleGAN: Conditional image-to-image translation is a separate application line, despite pix2pix's 23,338 citations; the PatchGAN discriminator it contributed is credited inside the VQGAN node.
- GigaGAN / StyleGAN-T: GAN text-to-image scale-ups that did not displace diffusion or AR; their MIT 6.S978 required-reading status is counted as evidence for the StyleGAN node.
- GMMN / MMD-GAN: Moment-matching precursors of Drifting Models with little adoption of their own.
- Glow / normalizing flows: A likelihood family that did not become an image-synthesis backbone; outside this roadmap's scope.
- VQGAN+CLIP: A 2021 community phenomenon rather than a method later work builds on; counted as evidence for the VQGAN and CLIP nodes.

**Curation notes:** Selection rule: high-confidence papers first, then papers that start or connect a line. The research kept 13 papers. DINOv2, the only medium one, is left out, so REPA is shown without arrows. CLIP and DALL-E were announced together on 2021-01-05, and the DALL-E paper reranks samples with CLIP, so CLIP is placed first. SimCLR, MoCo, Rotation Trick, Drifting Models and FD-Loss stay in the collection list.
