# Benchmarks

> **The canonical question for this chapter:**
> *What do benchmark scores actually measure  and why does a model that
> aces every standard benchmark sometimes fail badly in deployment?*

---

::: {.callout-note appearance="minimal"}
**Where are we?**


![The journey through the Model Mind.](figures/ch2/journey.svg){#fig-progress width="80%"}

@sec-human-eval established that human evaluation is the ground
truth and that everything else is an approximation. This chapter covers the
most widely used approximation: standardized benchmarks. They are fast, cheap,
reproducible, and consistently gameable  which is why understanding their
failure modes is as important as knowing their names.
:::

---

## What a Benchmark Is

A benchmark is a fixed dataset of evaluation examples with associated
scoring criteria that can be applied automatically, without human judgment,
to produce a comparable score across models. The input is a prompt or
a set of prompts; the output is a scalar or a small vector of scalars.
Two models evaluated on the same benchmark produce scores that can be
directly compared.

This comparability is the benchmark's primary virtue. Human evaluation
produces reliable comparisons but is slow, expensive, and cannot easily
be reproduced by independent parties. A benchmark can be run in hours
on any model, by anyone, and the results are directly comparable to
scores published by others running the same benchmark. This enables
systematic tracking of progress across the field: the score on MMLU
or HumanEval from a model released today can be compared to scores
from models released two years ago, providing a longitudinal view of
capability improvement.

The benchmark's primary vice follows from the same structure: because
benchmarks are fixed and public, they can be optimized against. A
model developer who knows the benchmark can collect training data that
overlaps with it, fine-tune on examples that resemble it, or select
model configurations that happen to perform well on its specific
format. The result is a benchmark score that reflects the model's
performance on that particular fixed test set, not the model's
general capability at the skill the benchmark was designed to measure.
This is Goodhart's Law applied to evaluation: when a measure becomes
a target, it ceases to be a good measure.

---

## A Taxonomy of Benchmarks

Language model benchmarks can be organized by what they measure and
how they measure it.

### Knowledge and Reasoning

**MMLU (Massive Multitask Language Understanding [@hendrycks2020measuring])**: 
57 academic subjects spanning STEM, humanities, social
sciences, law, and medicine. Each question is a four-way multiple
choice problem drawn from standardized tests, textbooks, and
academic courses. MMLU measures breadth of factual knowledge and
the ability to reason within academic domains.

MMLU scores: random baseline 25%, GPT-3.5 approximately 70%, GPT-4
approximately 86–87%, top models in 2024 above 89%. The benchmark
was designed when 60% was considered strong; saturation above 90%
has reduced its discriminative value for frontier models.

**GPQA (Graduate-Level Google-Proof Q&A [@rein2023gpqa])**: 448
multiple-choice questions in biology, chemistry, and physics, written
by domain experts to be answerable by experts but not by web search.
The "Google-proof" design attempts to prevent the benchmark from being
gamed by training on retrieved answers. Human expert performance is
approximately 65%; frontier models in 2024 score 50–60%, below human
expert level but above non-expert humans.

**ARC (AI2 Reasoning Challenge [@clark2018think])**: science
questions from U.S. elementary and middle school standardized tests.
Split into ARC-Easy (grade-school difficulty) and ARC-Challenge
(questions that retrieval systems fail at). ARC-Challenge was
a meaningful discriminator in 2019–2021; most frontier models now
exceed 90%.


### Mathematical Reasoning

**MATH [@hendrycks2021measuring]**: 12,500 competition mathematics
problems from AMC, AIME, and similar competitions, at difficulty
levels 1–5. Problems are open-ended (not multiple choice) and
require producing the correct numerical or algebraic answer.
Scoring: exact match on the final answer. GPT-4 without tools:
approximately 42–52%; with chain-of-thought: 60–70%; o1-class
models: above 90% on the full set.

**GSM8K [@cobbe2021training]**: 8,500 grade-school
arithmetic word problems requiring 2–8 reasoning steps. Much simpler
than MATH; most frontier models exceed 90% with chain-of-thought.
Remains useful for evaluating smaller models and for studying how
reasoning improves with scale and training.

**AIME (American Invitational Mathematics Examination)**: actual
competition problems from recent years, used informally as an
evaluation for frontier reasoning models. Not a standardized
benchmark with a canonical evaluation procedure, but widely
cited because competition mathematics problems are designed to
resist pattern matching and require genuine mathematical reasoning.

### Code Generation

**HumanEval [@chen2021evaluating]**: 164 handwritten Python programming
problems, each with a function signature, docstring, and test suite.
Scored by pass@k: the probability that at least one of $k$ generated
samples passes all unit tests. HumanEval measures basic coding ability
for well-defined algorithmic tasks. Frontier models now exceed 90%
pass@1, near saturation.

**MBPP (Mostly Basic Python Programming, [@austin2021program])**: 374
crowd-sourced Python problems at a range of difficulty levels. Similar
structure to HumanEval but with more diverse problem types including
data manipulation, string processing, and simple algorithms.

**SWE-bench [@jimenez2024swe]**: real GitHub issues from popular
Python open-source repositories. The model must produce a patch that
resolves the issue, evaluated by running the repository's test suite.
SWE-bench measures agentic software engineering capability (reading
existing code, understanding bug reports, producing correct patches)
rather than isolated algorithmic problem solving. Much harder than
HumanEval: the best models in mid-2024 resolve 10–20% of issues on
the verified subset (SWE-bench Verified).

### Language Understanding and NLU

**SuperGLUE [@wang2019superglue]**: a collection of language
understanding tasks including question answering, natural language
inference, coreference resolution, and word sense disambiguation.
Designed as a harder successor to GLUE; frontier models exceed
human baseline on most tasks. Largely obsolete as a discriminator
for current models.

**BIG-Bench [@srivastava2022beyond]**: 204 tasks covering an
intentionally diverse range of capabilities including logic, arithmetic,
common sense, translation, and novel task formats. BIG-Bench Hard
selects the 23 tasks where models
underperformed humans at the time of publication. Remains a useful
evaluation for unusual capabilities not covered by narrower benchmarks.

### Instruction Following

**MT-Bench [@zheng2023judging]**: 80 multi-turn questions across
eight categories (writing, roleplay, reasoning, math, coding, extraction,
STEM, humanities). Evaluated by GPT-4 as judge which Measures
instruction following across diverse topics and conversation structures.

**IFEval [@zhou2023instruction]**: instruction following evaluation with
verifiable constraints, prompts that require following explicit formatting
instructions (use bullet points, respond in French, limit to 100 words).
Scored automatically by checking whether the specified constraints are met,
providing a cleaner signal than human or LLM judgment for the specific
capability of following explicit formatting instructions.

---

## Saturation: When Benchmarks Stop Discriminating

A benchmark saturates when most evaluated models achieve scores close
to the ceiling, eliminating its discriminative value. Saturation is a
natural consequence of progress: a benchmark designed to be challenging
in 2020 may be trivial for 2024 models.

The saturation timeline for major benchmarks:

| Benchmark | Release | Frontier model performance | Saturation status |
|-----------|---------|--------------------------|------------------|
| HellaSwag | 2019 | >95% | Saturated |
| ARC-Challenge | 2018 | >90% | Saturated |
| SuperGLUE | 2019 | >90% | Saturated |
| GSM8K | 2021 | >95% | Saturated |
| HumanEval | 2021 | >90% | Saturated |
| MMLU | 2021 | ~89% | Approaching saturation |
| MATH | 2021 | ~90% (o1) | Task-dependent |
| GPQA | 2023 | ~60% | Not saturated |
| SWE-bench | 2024 | ~20% | Not saturated |

The pattern is consistent: a benchmark is published when frontier models
score roughly 30–60%; models reach human performance within 1–3 years;
the benchmark stops being informative for frontier model comparison
shortly after. The half-life of a discriminative benchmark has shortened
as model capabilities advance: benchmarks that were useful for 3–4 years
in the 2019–2021 period have been useful for 1–2 years in the 2022–2024
period.

The field's response is to continuously develop harder benchmarks, which
frontier models then approach the ceiling of, requiring still harder
benchmarks. This is a healthy response to capability progress but makes
longitudinal comparison across benchmark generations difficult: a score
on GPQA cannot be directly compared to a score on MMLU because the
difficulty distributions are entirely different.

---

## Contamination: The Primary Validity Threat

Benchmark contamination (the presence of benchmark test examples in a
model's training data) is the most serious threat to benchmark validity.
A model that has seen the test examples during training can achieve a
high score through memorization rather than the capability the benchmark
is designed to measure.

### How Contamination Occurs

Web-scale training corpora contain enormous fractions of the publicly
available internet, including benchmark datasets that were posted publicly.
MMLU, HumanEval, GSM8K, and most other major benchmarks have been
publicly available since their release and have likely appeared in the
training corpora of models trained on large web crawls.

Contamination can be unintentional (the benchmark was included in a
web crawl without the model developer realizing it) or intentional
(the model developer deliberately included benchmark-adjacent data to
improve scores). The distinction matters ethically but not practically:
the benchmark score is inflated either way.

### Detecting Contamination

Several methods exist for detecting whether a model has seen specific
benchmark examples during training:

**N-gram overlap**: check whether high-n-gram sequences from the benchmark
appear in the training corpus. If the training corpus is available for
inspection (which is often not), this is straightforward. If not,
the model can be probed: ask the model to complete the latter half of
a benchmark example; high completion accuracy suggests memorization.

**Verbatim completion probing**: present the model with the first few
tokens of a benchmark question and measure whether it accurately completes
the rest. Golchin and Surdeanu [@golchin2024time] used this method to detect
contamination across multiple benchmarks, finding significant evidence
of contamination in several major models.

**Performance vs. difficulty correlation**: if a model performs uniformly
well across easy and hard examples within a benchmark (rather than showing
higher accuracy on easier examples), this is a signature of memorization,
a model that has memorized answers does not exhibit the difficulty sensitivity
that genuine comprehension would produce.

**Held-out test sets**: the most reliable contamination prevention is to maintain
a private test set that is never released publicly. This is the approach
used by ML competitions (Kaggle, SWE-bench Verified) but is difficult for
academic benchmarks because the research community needs access to the test
set to evaluate their own models.

### The Contamination Response

When contamination is detected or suspected, the standard responses are:

1. **Reproduce evaluation with contamination-filtered test sets**: remove
   examples that overlap with the training corpus and re-evaluate. The
   performance drop (if any) quantifies the contamination effect.

2. **Report contamination alongside scores**: some model technical reports
   include contamination analysis; the field increasingly expects this.

3. **Create new evaluation versions**: update the benchmark with new
   examples that are not in existing training corpora. GSM8K's authors
   released GSM-Symbolic which generates novel variants
   of grade-school math problems that cannot have been memorized.

4. **Use dynamic or private benchmarks**: evaluation on private test sets
   or dynamically generated problems (Chatbot Arena's real-user prompts
   are a natural example) is inherently contamination-resistant.

---

## Format Sensitivity and Prompt Brittleness

Benchmark scores are sensitive to how prompts are formatted. The same
model, evaluated on the same benchmark with different prompt formats,
can show performance differences of 5–15 percentage points which is
large enough to change the apparent ranking between models.

### Few-Shot Format Effects

Most benchmarks are evaluated in a few-shot setting: 0–5 examples of
correctly answered questions are prepended to the test question. The
number of few-shot examples, their selection, and their format all
affect performance:

**Zero-shot vs. few-shot**: adding 5-shot examples improves performance
by 3–10 points on most knowledge benchmarks for models that were not
specifically instruction-tuned. Instruction-tuned models sometimes perform
better zero-shot than few-shot, because the few-shot examples interfere
with the model's learned instruction-following behavior.

**Example selection**: randomly selected few-shot examples produce
different scores than examples selected for difficulty, diversity,
or similarity to the test question. Benchmark reporting conventions
specify a standard selection procedure, but not all evaluations
follow the same convention.

**Chain-of-thought prompting**: adding "Let's think step by step" or
few-shot examples with reasoning chains substantially improves performance
on reasoning benchmarks (5–30 points on MATH) and may reduce performance
on simple recall benchmarks where the reasoning is counterproductive.
Whether CoT is used must be specified in benchmark reporting.

### Answer Format Sensitivity

Multiple-choice benchmarks can be answered by producing the letter
(A, B, C, D) or by producing the full text of the correct answer.
Models have different preferences for these formats and may perform
differently depending on which is requested. Instruction-tuned models
often perform better when asked to produce the full answer text rather
than just the letter, because instruction tuning teaches verbose responses.

Some evaluations use log-probability scoring rather than generation:
the model is not asked to generate the answer but to assign probabilities
to each candidate answer; the highest-probability candidate is selected.
Log-probability scoring eliminates format sensitivity but introduces
sensitivity to tokenization: candidates that tokenize into more tokens
may receive systematically lower log-probabilities simply because the
probability is distributed across more tokens (each multiplied, resulting
in a lower joint probability).

---

## The Leaderboard Problem

Benchmark leaderboards (public rankings of models by benchmark score)
have become a primary mechanism for communicating model quality to
practitioners and the public. They are also a mechanism that concentrates
optimization pressure in ways that degrade benchmark validity.

### Competitive Pressure and Gaming

When benchmark scores become the primary signal of model quality, model
developers have strong incentives to optimize specifically for high
benchmark scores. Strategies that improve benchmark scores without
improving general capability:

**Benchmark-specific fine-tuning**: fine-tune on data that resembles
the benchmark, improving performance on that benchmark's specific
format without improving general capability.

**Benchmark-specific prompt engineering**: develop prompts specifically
calibrated for each benchmark's format, which may not generalize to
other evaluation contexts.

**Selective reporting**: publish scores only on benchmarks where the
model performs well; omit or de-emphasize benchmarks where it performs
poorly. A model report that includes MMLU, GSM8K, and HumanEval scores
but not GPQA or SWE-bench is not necessarily hiding anything, but the
selection shapes the impression of overall capability.

**Hyperparameter selection on test sets**: tune decoding parameters
(temperature, top-p, beam width) on the benchmark test set rather than
a development set. This is a form of overfitting to the benchmark without
training on it.

### Benchmark Proliferation

The response to gaming (developing new, harder benchmarks) has
produced an enormous number of benchmarks. A model technical report
in 2024 may include scores on 20–50 different benchmarks. This
proliferation creates its own problems:

Readers cannot interpret 50 benchmark scores simultaneously. The
effective communication of model capability requires either aggregation
(combining scores into a single number) or selection (choosing a small
number of representative benchmarks) both of which introduce their
own distortions.

---

## What Benchmarks Miss

Even well-designed, uncontaminated benchmarks with careful evaluation
protocols miss important dimensions of model quality that matter in
deployment.

### Long-Horizon Tasks

Most benchmarks evaluate single-turn responses to short prompts. Real
deployments involve multi-turn conversations, long-document processing,
and tasks requiring sustained coherent effort over many steps. A model
that scores well on MMLU's multiple-choice questions may produce
incoherent output when asked to write a 5,000-word technical report
or maintain consistency across a 50-turn conversation.

SWE-bench is a partial exception: resolving a GitHub issue requires
reading a codebase, understanding the bug report, producing a patch,
and having the patch pass tests is a multi-step task with real-world
structure. Its difficulty (20% resolution rate at the frontier) partly
reflects this structural difference from single-turn question answering.

### Calibration

Benchmarks measure accuracy but not calibration. A model that is right
80% of the time and knows when it is wrong (expressing uncertainty on
the 20% it gets wrong) is far more useful than a model that is right
80% of the time and is equally confident on all answers. Standard
benchmark scoring does not distinguish these models.

### Novel Task Formats

Benchmarks test performance on tasks in the format defined by the
benchmark creators. Real users present tasks in idiosyncratic formats,
with implicit context, ambiguous specifications, and novel combinations
of requirements that no benchmark anticipated. A model's generalization
to novel task formats is not measured by any existing benchmark and
requires the human evaluation.

### Robustness

Benchmark scores reflect average performance on the test set. A model
that achieves 85% accuracy by being correct on 85% of examples but
completely wrong on 15% may be less useful than a model that achieves
80% accuracy by being approximately right on all examples. The distribution
of errors including their severity, correlation with input features, or
predictability matters enormously in deployment and is not captured
by accuracy scores.



---

## Key Takeaways

- Benchmarks are fixed evaluation datasets with automated scoring that
  enable fast, reproducible, comparable measurement; their primary vice
  is that fixed public benchmarks can be optimized against, making scores
  reflect benchmark-specific performance rather than general capability.
- The saturation timeline for major benchmarks has shortened: HellaSwag,
  ARC, GSM8K, and HumanEval are saturated for frontier models; MMLU is
  approaching saturation; GPQA and SWE-bench remain discriminative as of
  2024.
- Benchmark contamination — test examples appearing in training data —
  is the most serious validity threat; detection methods include n-gram
  overlap analysis, verbatim completion probing, and performance-difficulty
  correlation analysis.
- Format sensitivity produces 5–15 point performance differences from
  prompt wording alone; few-shot count, chain-of-thought, and answer
  format must all be specified and held constant for valid comparison.
- Log-probability scoring eliminates generation format sensitivity but
  introduces tokenization sensitivity; neither scoring method is universally
  superior.
- Benchmark leaderboard competition concentrates optimization pressure on
  benchmark scores, producing benchmark-specific fine-tuning, selective
  reporting, and hyperparameter selection on test sets — all of which inflate
  scores without improving deployment capability.
- Benchmarks systematically miss long-horizon tasks, calibration, novel
  task format generalization, and error distribution — all of which matter
  in deployment and require human evaluation or specialized instruments.
- Responsible benchmark use requires: multiple benchmarks, full reporting
  of all benchmarks run, contamination analysis, protocol specification,
  and complementary human evaluation for release decisions.
- The correct interpretation of a benchmark score is "performance on this
  specific fixed test set under this specific evaluation protocol" — never
  "general capability at the skill the benchmark was designed to measure."

---

## Further Reading

- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D.,
  & Steinhardt, J. (2021). *Measuring Massive Multitask Language
  Understanding.* ICLR. — Introduces MMLU; the 57-subject design rationale
  and the baseline analysis across model families establish the template
  for broad knowledge evaluation benchmarks.

- Srivastava, A., et al. (2022). *Beyond the Imitation Game: Quantifying
  and Extrapolating the Capabilities of Language Models.* TMLR. —
  BIG-Bench; the 204-task diversity, the emergent capability analysis,
  and the BIG-Bench Hard subset selection are the key contributions;
  the discussion of what tasks resist scaling is particularly valuable.

- Chen, M., et al. (2021). *Evaluating Large Language Models Trained on
  Code.* arXiv. — Introduces HumanEval and the pass@k metric; the
  functional correctness evaluation methodology and the analysis of
  temperature's effect on pass@k are the key technical contributions.

- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O.,
  & Narasimhan, K. (2024). *SWE-bench: Can Language Models Resolve
  Real-World GitHub Issues?* ICLR. — Introduces SWE-bench; the real-world
  GitHub issue structure and the automated test suite evaluation are the
  key design contributions; the analysis of why existing models fail is
  the most informative section for practitioners.

- Golchin, S., & Surdeanu, M. (2023). *Time Travel in LLMs: Tracing
  Data Contamination in Large Language Models.* ICLR. — Systematic
  contamination detection using completion probing; the methodology is
  the key contribution; the contamination rates reported across major
  models and benchmarks are sobering.

- Liang, P., et al. (2022). *Holistic Evaluation of Language Models.*
  TMLR. — HELM; the multi-scenario, multi-metric evaluation framework
  and the analysis of trade-offs between accuracy, calibration, and
  robustness are the key contributions; the selective reporting analysis
  documents how benchmark choice affects apparent rankings.

- Rein, D., et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A
  Benchmark.* arXiv. — Introduces GPQA and the Google-proof design
  methodology; the expert versus non-expert performance gap and the
  analysis of what makes questions genuinely hard are the key contributions.

- Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). *Do
  ImageNet Classifiers Generalize to ImageNet?* ICML. — Though not
  an NLP paper, this is the clearest demonstration that held-out
  test sets degrade as proxies for general performance when the
  field optimizes against them; the methodology and conclusions
  transfer directly to language model benchmarking.

---
