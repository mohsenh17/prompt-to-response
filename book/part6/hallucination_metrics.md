# Hallucination Metrics

> **The canonical question for this chapter:**
> *How do you automatically detect when a language model has generated
> something that is confidently stated but factually wrong and why is
> this a harder measurement problem than it first appears?*

---

::: {.callout-note appearance="minimal"}
**Where are we?**

![The journey through the Model Mind.](figures/ch6/journey.svg){#fig-progress width="80%"}


The previous two chapters covered metrics that measure
output quality by comparing it to a reference n-gram overlap and embedding
similarity. Both fail on the same dimension: factual accuracy. A fluent,
well-organized response that confidently states false information scores
well on BLEU and reasonably on BERTScore. This chapter covers metrics
designed specifically to detect that failure mode.
:::

---

## Defining Hallucination

The word "hallucination" entered NLP evaluation vocabulary to describe a
specific failure mode of language models: generating confident, fluent,
grammatically correct statements that are factually wrong or unsupported
by the available evidence. The term is borrowed from clinical psychology,
where hallucinations are perceptions without external stimuli, things that
feel real but are not. In language models, the analogy holds: the model
produces outputs that feel correct, coherent, authoritative, and plausible
but are detached from reality.

Hallucination takes several distinct forms that require different detection
approaches:

**Intrinsic hallucination**: the generated text contradicts, distorts, 
or misrepresents information contained in the input. In summarization, 
if the source states that a drug reduced mortality but did not affect 
disease progression, and the summary claims that the drug improved progression, 
the summary contains an intrinsic hallucination. Because the source provides 
the reference, detection can focus on whether the generated claims are consistent 
with the input.

**Extrinsic hallucination**: the generated text introduces claims 
that are not supported or entailed by the input. These claims may be 
true or false according to external knowledge. In summarization, adding 
a detail such as a specific treatment cost, date, or event that is absent 
from the source is an extrinsic hallucination. Determining whether such 
claims are actually true generally requires external evidence or knowledge 
beyond the input.

**Entity hallucination**: the model generates an incorrect, nonexistent, 
or unsupported entity or entity attribute. Examples include inventing a 
person's name, citing a nonexistent paper, assigning an event to the wrong 
date, or referring to a place that does not exist. Entity hallucinations are 
particularly consequential in applications where users rely on precise names, 
identifiers, dates, or citations.

**Factual inconsistency**: the model generates a claim that conflicts with 
established facts or reliable external knowledge. For example, a model might 
correctly identify a person but incorrectly state their occupation, affiliation, 
or historical role. Unlike intrinsic hallucination, which concerns inconsistency 
with the input, factual inconsistency is evaluated against facts about the world.

**Reasoning errors**: the model reaches an incorrect conclusion because one or 
more inference steps are invalid, even when the underlying facts or premises may 
be correct. For example, a model may correctly identify that A is greater than B 
and B is greater than C but incorrectly conclude that C is greater than A. Detecting 
reasoning errors requires evaluating the validity of each reasoning step, not just the 
conclusion.


The appropriate metric depends on which hallucination type is of concern.
No single metric detects all types reliably.

---

## Why Standard Metrics Miss Hallucination

The failure of BLEU and BERTScore to detect hallucinations is not accidental
it is a direct consequence of what these metrics measure.

BLEU measures lexical overlap. A hallucination that shares vocabulary with
the reference ("The experiment was conducted in 1984 in London" vs. reference
"The experiment was conducted in 1984 in Paris") scores well on BLEU because
most n-grams match. The factually critical word ("London" vs. "Paris") is
a single token that differs; the rest of the n-gram structure is identical.

BERTScore measures semantic similarity in embedding space. The embeddings of 
"London" and "Paris" are semantically close both are European capitals, both 
appear in similar contexts in training data. The embedding distance between a 
hallucinated city name and the correct one may be smaller than the embedding 
distance between two semantically distinct words, making BERTScore insensitive 
to exactly the factual distinctions that matter most.

Perplexity is insensitive by design: a hallucinated fact stated fluently
may be assigned higher probability than the correct fact stated awkwardly,
because the model has learned to assign probability based on surface plausibility
rather than factual accuracy.

---

## Faithfulness vs. Factuality

A critical distinction separates two different hallucination evaluation
problems:

**Faithfulness**: does the output stay faithful to the provided source
material? A summarization system that introduces information not in the
source document has a faithfulness problem, independent of whether that
information is true according to external knowledge. Faithfulness is
evaluable without external knowledge and the source document provides the
reference.

**Factuality**: does the output accurately reflect world knowledge? A
question-answering system that generates a plausible but incorrect answer
has a factuality problem. Factuality evaluation requires external knowledge
or verification against a knowledge source.

Most automatic hallucination metrics evaluate faithfulness rather than
factuality, because faithfulness is tractable: the source document is
available as a reference. Factuality evaluation at scale is significantly
harder because it requires either: a comprehensive knowledge base covering
the claims the model might make, or another model that can verify claims 
which introduces its own accuracy and coverage limitations.

---

## Natural Language Inference as Hallucination Detection

The earliest systematic approach to faithfulness evaluation repurposed
Natural Language Inference (NLI) models. NLI is the task of determining
whether a hypothesis is entailed by a premise, contradicted by it, or
neutral with respect to it. Given a source document (premise) and a
generated sentence (hypothesis), an NLI model can classify whether the
generated sentence is:

- **Entailed**: the source supports the generated claim
- **Contradicted**: the source contradicts the generated claim
- **Neutral**: the source neither supports nor contradicts the claim
  (the claim introduces information not in the source)

Entailed sentences are faithful; contradicted sentences are intrinsic
hallucinations; neutral sentences are extrinsic hallucinations.

### FactCC and Related Models

FactCC [@kryscinski2020evaluating] is a BERT-based NLI model fine-tuned
specifically to detect factual consistency between a source document
and a generated summary. Rather than using standard NLI training data
(which is drawn from human-written premises and hypotheses), FactCC
was trained on synthetically generated (document, summary) pairs with
controlled factual errors: entity swaps, number changes, pronoun changes,
negation insertion.

The training procedure: take a human-written summary, apply a controlled
transformation (substitute the correct person's name with another person's
name from the document), and label the transformed pair as factually
inconsistent. The original pair is labeled consistent. Fine-tuning on
these synthetic examples teaches FactCC to detect specific types of
factual errors without requiring large-scale human annotation of
hallucinations.

FactCC achieves approximately 75–80% accuracy at detecting factual
inconsistency in CNN/DailyMail summaries, compared to human annotator
accuracy of approximately 85–90% on the same task. The 10–15% gap reflects
FactCC's limitation: it detects the types of errors it was trained on
(entity swaps, number errors) better than novel error types.


---

## Question-Answering Based Hallucination Detection

A separate family of faithfulness metrics uses question generation and
answering to evaluate consistency. The core idea: if a generated summary
is faithful to the source document, questions about the summary should
be answerable from the source document, with matching answers.

### QAGS: Question Answering and Generation for Summarization

QAGS [@wang2020asking] operates as follows:

1. **Generate questions** from the hypothesis (generated summary). A
   question generation model takes each sentence of the summary and
   generates factual questions about it: "Who conducted the experiment?",
   "When did the event occur?", "What was discovered?"

2. **Answer questions** from two sources: the hypothesis itself and the
   source document. A question answering model answers each question twice,
   once using the summary and once using the source document.

3. **Compare answers**. If the answer from the summary matches the answer
   from the source document, that claim in the summary is faithful. If they
   differ, the summary has introduced information not supported by the source.

4. **Aggregate**. The QAGS score is the fraction of questions where the
   summary and source answers match, measuring overall faithfulness.

QAGS achieves higher correlation with human faithfulness judgments than
NLI-based methods on summarization tasks. Its advantage: it is more robust
to the specific error types present in the generated text, because the
questions are generated from the specific content being evaluated rather
than relying on a classifier trained on pre-specified error types.

Its limitations: question generation quality affects reliability (poorly
formed questions produce unreliable answers), and answering questions from
long source documents is itself an imperfect process prone to errors.

---

## FActScoring: Decomposed Factuality Evaluation

FactScore [@min2023factscore] addresses the problem of evaluating factuality 
in long-form text generation, focusing not just on faithfulness to a source 
but also on accuracy with respect to world knowledge. It is the most widely 
adopted metric for evaluating factual accuracy in knowledge-intensive generation
tasks.

### The Decomposition Approach

FActScore's key insight: rather than evaluating an entire generated
response for factual accuracy, decompose it into individual atomic facts
and verify each separately. This decomposition:

1. Produces more granular, actionable feedback (which specific claims
   are wrong, not just an overall score)
2. Allows verification of each claim independently using a dedicated
   retrieval and verification system
3. Scales gracefully to long documents where holistic evaluation is
   difficult

**Step 1: Atomic fact extraction.** A language model
decomposes the generated text into a list of atomic facts which are minimal,
self-contained claims that can each be independently verified. For
the sentence "Marie Curie, who was born in Warsaw in 1867, won two
Nobel Prizes in different scientific fields", the atomic facts might be:

- Marie Curie was born in Warsaw
- Marie Curie was born in 1867
- Marie Curie won two Nobel Prizes
- Marie Curie's Nobel Prizes were in different scientific fields

**Step 2: Fact verification.** Each atomic fact is verified against a
knowledge source (typically Wikipedia) using a combination of retrieval
and NLI:

- Retrieve the most relevant Wikipedia passages for the claim
- Use an NLI model to classify each claim as supported, refuted, or
  not enough information given the retrieved passages

**Step 3: FActScore computation.** The FActScore is the fraction of
atomic facts that are supported by the knowledge source:

$$
\text{FActScore} = \frac{\text{\# supported facts}}{\text{\# total atomic facts}}
$$

### FActScore Limitations

**Atomic fact extraction quality**: the decomposition step depends on a
language model to identify atomic facts, which itself introduces errors.
Claims that are difficult to decompose, implicit claims, or claims with
complex dependencies may be extracted incorrectly or missed.

**Knowledge source coverage**: FActScore uses Wikipedia as its
verification source. Claims about topics not well-covered in Wikipedia
(recent events, specialized domains, non-English subjects) receive
"not enough information" rather than being correctly verified.
The metric scores these as unsupported, which may overestimate
hallucination rates for legitimate claims that Wikipedia does not cover.

**Verification model accuracy**: the NLI model used for verification
makes its own errors. Claims that are true but stated in ways that
differ from Wikipedia's phrasing may receive incorrect "not supported"
labels; claims that happen to match Wikipedia's phrasing superficially
may receive incorrect "supported" labels despite factual differences.



---


## Key Takeaways

- Hallucination takes several distinct forms — intrinsic (contradicts
  input), extrinsic (introduces unsupported information), entity
  (wrong named entities), and reasoning errors — each requiring different
  detection approaches.
- BLEU and BERTScore fail to detect hallucinations because they measure
  lexical or semantic overlap, not factual accuracy; a single wrong entity
  in an otherwise correct response may barely affect these scores.
- Faithfulness evaluation (does the output follow from the source?) is
  tractable using NLI or QA-based methods; factuality evaluation (is the
  output true according to world knowledge?) is harder and requires a
  knowledge source.
- NLI-based metrics (FactCC) achieve approximately 75–80% accuracy at
  detecting factual inconsistency in summarization; their limitation is
  sensitivity to the specific error types present in their training data.
- QA-based metrics (QAGS, QAFactEval) generate questions from the
  hypothesis and compare answers from hypothesis and source, achieving
  higher correlation with human judgments (Pearson ~0.45–0.55) than NLI
  methods on summarization.
- Retrieval-augmented generation reduces hallucination substantially:
  FActScore of ~80% for retrieval-augmented systems versus ~58–73% for
  non-retrieval systems, directly motivating the RAG architecture.
- Expressed model confidence is a weak hallucination signal; calibration
  is insufficient for reliable detection, with the best confidence
  elicitation strategies achieving AUC-ROC of 0.70–0.75 at distinguishing
  correct from hallucinated claims.


---

## Further Reading

- Kryscinski, W., McCann, B., Xiong, C., & Socher, R. (2020).
  *Evaluating the Factual Consistency of Abstractive Text Summarization.*
  EMNLP. — Introduces FactCC and the synthetic training data generation
  procedure; the controlled perturbation approach to creating training
  examples is the key methodological contribution; the analysis of
  which error types FactCC detects best reveals its scope and limitations.

- Wang, A., Cho, K., & Lewis, M. (2020). *Asking and Answering Questions
  to Evaluate the Factual Consistency of Summaries.* ACL. — Introduces
  QAGS; the question generation and dual-source answering procedure is
  the key contribution; the human correlation analysis establishes QAGS
  as more reliable than NLI-based methods on summarization.

- Fabbri, A. R., Wu, C.-S., Liu, W., & Xiong, C. (2022). *QAFactEval:
  Improved QA-Based Factual Consistency Evaluation for Summarization.*
  NAACL. — Improves QAGS with stronger component models and learned
  aggregation; the human correlation comparison across methods on multiple
  summarization benchmarks is the key empirical contribution.

- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W.,
  Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). *FActScoring:
  Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text
  Generation.* EMNLP. — Introduces FActScore and the atomic fact
  decomposition pipeline; the biography generation evaluation and the
  comparison across model families are the key empirical contributions;
  the Wikipedia retrieval verification pipeline is described precisely
  enough to reproduce.

- Goyal, T., & Durrett, G. (2020). *Evaluating Factuality in Generation
  with Dependency-Level Entailment.* EMNLP Findings. — Introduces DAE
  and dependency-arc-level faithfulness evaluation; the fine-grained
  localization of hallucinated content is the key contribution relative
  to document-level NLI approaches.

- Kadavath, S., et al. (2022). *Language Models (Mostly) Know What They
  Know.* arXiv. — Studies self-knowledge calibration in language models;
  the finding that models show some but insufficient self-knowledge to
  reliably identify their own hallucinations motivates external verification
  approaches; the calibration curves are the key empirical contribution.

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).
  *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv.
  — Introduces RAGAS and the faithfulness, answer relevance, context
  precision, and context recall metrics for RAG evaluation; the decomposition
  of RAG quality into retrieval and generation components is the key
  contribution for practitioners building RAG systems.

- Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B. (2024).
  *Can LLMs Express Their Uncertainty? An Empirical Evaluation of
  Confidence Elicitation in LLMs.* ICLR. — Systematic evaluation of
  confidence elicitation strategies; the AUC-ROC analysis across
  strategies and model families is the key empirical contribution;
  the finding that no strategy reliably identifies hallucinations is
  the key practical conclusion.

---