# Lexical Metrics

> **The canonical question for this chapter:**
> *What do n-gram overlap metrics actually measure, why did they dominate
> NLP evaluation for two decades, and why are they now inadequate as the
> primary measure of language model output quality?*

---

::: {.callout-note appearance="minimal"}
**Where are we?**

![The journey through the Model Mind.](figures/ch4/journey.svg){#fig-progress width="80%"}


Perplexity measures how well a model predicts its training
distribution. Lexical metrics (BLEU, ROUGE, and METEOR) measure something 
different: how much a generated output overlaps with a reference output. This 
chapter covers the n-gram overlap family, how each metric is computed, what 
it was designed to measure, where it works, and why it fails for the open-ended 
generation tasks that dominate current language model deployment.
:::

---

## The Reference-Based Evaluation Paradigm

Perplexity requires no reference, it is a property of the model alone,
measuring how well it predicts any text. BLEU, ROUGE, and METEOR belong
to a different paradigm: reference-based evaluation. They require a gold
standard (a human-written reference output) and measure how closely the
model's generated output matches it.

Reference-based evaluation was the natural approach for NLP tasks with
constrained output spaces. Machine translation has a correct target: a
good English translation of a German sentence. Summarization has a
reference: a human-written summary of the same document. When the reference
captures what a good output looks like, measuring similarity to the reference
is a reasonable proxy for output quality.

The limitation surfaces immediately: most tasks have many correct outputs,
not one. A German sentence can be correctly translated into English in multiple
ways that differ in word choice, sentence structure, and toneand all of
them are good and yet different from each other. A document can be summarized
in many valid ways. A question can be answered correctly with different
phrasings. When the output space is large and diverse, a reference-based
metric that rewards similarity to a single reference penalizes good outputs
that differ from the reference in arbitrary ways.

This limitation was understood from the beginning. BLEU  was proposed with 
multiple reference translations, averaging similarity across several human-written 
references to partially address the single-reference problem. But multiple 
references are expensive to collect, and even with four or five references, the 
space of acceptable outputs is far larger than the references cover.

For open-ended generation (the dominant mode of language model use) there
is often no meaningful reference at all. A user asks a language model to
explain quantum entanglement. What is the reference? Any of hundreds of
valid explanations, at various levels of detail, with various analogies,
in various tones. Reference-based evaluation is structurally ill-suited
to this regime.

---

## BLEU: Bilingual Evaluation Understudy

BLEU (Bilingual Evaluation Understudy) was developed at IBM to automatically 
evaluate machine translation output. It became the standard MT evaluation metric 
for nearly two decades and remains one of the most widely cited metrics in NLP despite 
well-documented limitations.

### The Computation

BLEU computes modified n-gram precision: the fraction of n-grams in the
hypothesis (generated output) that appear in any of the reference translations,
clipped by the count of that n-gram across all references.

For unigrams: count the number of words in the hypothesis that appear in
the references, with each reference word counting at most once per hypothesis
word. For bigrams: count consecutive word pairs. For trigrams and 4-grams
similarly.

The modified precision for n-grams:

$$
p_n = \frac{\sum_{C \in \text{hypothesis}} \sum_{n\text{-gram} \in C} \text{Count}_{\text{clip}}(n\text{-gram})}
{\sum_{C \in \text{hypothesis}} \sum_{n\text{-gram} \in C} \text{Count}(n\text{-gram})}
$$

where $\text{Count}_{\text{clip}}$ caps the count of each n-gram at its
maximum count in any single reference.

BLEU then combines n-gram precisions across n = 1, 2, 3, 4 using a geometric
mean, and applies a brevity penalty to prevent the metric from being gamed
by extremely short hypotheses (which would have high precision simply by
avoiding any words not in the reference):

$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)
$$

where $w_n = 1/4$ (equal weights for 1-gram through 4-gram precision) and
the brevity penalty is:

$$
\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}
$$

with $c$ the length of the hypothesis and $r$ the length of the closest
reference. A hypothesis shorter than the reference is penalized; a hypothesis
longer than the reference is not.

### A Worked Example

Reference: "The cat sat on the mat."
Hypothesis A: "The cat sat on the mat."
Hypothesis B: "The cat is on the mat."
Hypothesis C: "A feline rested upon the rug."

Unigram precision:
- A: 6/6 = 1.0 (all words match)
- B: 5/6 = 0.83 ("is" does not appear in reference)
- C: 1/6 = 0.17 ("the" matches and the rest don't)

Bigram precision for C: none of the bigrams in C appear in the reference
("A feline", "feline rested", "rested upon", "upon the", "the rug") — 0/5 = 0.

Hypothesis C would receive a very low BLEU score despite being a semantically
valid paraphrase. This is the n-gram overlap metric's fundamental problem
with synonymy: it rewards lexical similarity, not semantic equivalence.

### What BLEU Measures and Does Not Measure

BLEU measures **lexical overlap** between hypothesis and reference. It does
not measure:

- **Semantic equivalence**: synonyms and paraphrases are penalized
- **Grammatical correctness**: an ungrammatical hypothesis that copies many
  reference n-grams may score higher than a grammatical hypothesis that
  paraphrases
- **Factual accuracy**: a hypothesis that copies reference words while
  asserting the wrong facts scores similarly to one asserting correct facts
- **Fluency**: a hypothesis with correct words in scrambled order receives
  some n-gram credit at the unigram level

The correlation between BLEU scores and human judgments was the original
motivation for the metric. For machine translation evaluated with multiple
references at the corpus level (not sentence level), BLEU showed reasonable
correlation with human rankings of translation systems in 2002. This
correlation has been repeatedly challenged as models improved, translation
outputs diversified, and the task distribution shifted beyond the narrow
news translation domain where BLEU was calibrated.

### BLEU's Sensitivity to Corpus vs. Sentence Level

BLEU was designed for corpus-level evaluation: compute the metric over
a large test set of sentences and compare systems at the aggregate level.
At the sentence level, BLEU is unreliable, a single sentence's n-gram
overlap is too noisy to be informative. A hypothesis that differs from
the reference in a single word may receive a very different BLEU score
from one that differs in a paraphrase of the entire sentence, even if
human judges rate them equivalently.


---

## ROUGE: Recall-Oriented Understudy for Gisting Evaluation

ROUGE was developed for evaluating automatic summarization.
Where BLEU measures precision (how much of the hypothesis appears in the
reference), ROUGE emphasizes recall (how much of the reference appears
in the hypothesis), reflecting the summarization task's concern with
coverage, did the summary include the important information from the
source document?

### ROUGE Variants

**ROUGE-N**: n-gram recall between hypothesis and reference:

$$
\text{ROUGE-N} = \frac{\sum_{S \in \text{references}} \sum_{n\text{-gram} \in S} \text{Count}_{\text{match}}(n\text{-gram})}
{\sum_{S \in \text{references}} \sum_{n\text{-gram} \in S} \text{Count}(n\text{-gram})}
$$

ROUGE-1 (unigram recall) and ROUGE-2 (bigram recall) are the most commonly
reported. ROUGE-1 is sensitive to shared vocabulary; ROUGE-2 is sensitive
to shared phrase structure.

**ROUGE-L**: Longest Common Subsequence (LCS) between hypothesis and
reference, normalized by the length of the reference. LCS-based recall
captures sentence-level word order similarity more sensitively than n-gram
recall, because the longest common subsequence must appear in order (though
not necessarily contiguously). ROUGE-L is more robust to paraphrase than
ROUGE-N but still penalizes synonymy.


---

## METEOR: Metric for Evaluation of Translation with Explicit ORdering

METEOR was designed to address BLEU's two
most obvious limitations: its lack of recall and its inability to match
synonyms. METEOR incorporates both precision and recall and extends the
matching to include stemming and synonymy through WordNet.

### The Computation

METEOR alignment proceeds in stages:

1. **Exact match**: find words in hypothesis and reference that match exactly.
2. **Stem match**: find words that share the same stem (e.g., "running"
   matches "run").
3. **Synonym match**: find words that are synonyms in WordNet (e.g.,
   "automobile" matches "car").

These matches are found greedily, prioritizing exact matches over stem matches
over synonym matches. The matched words define an alignment between hypothesis
and reference.

From the alignment, precision $P$ and recall $R$ are computed over the matched
words. An F-mean is computed with recall weighted more heavily than precision
(typical weight: precision weight = 1, recall weight = 9, giving recall
9× more weight):

$$
F_{\text{mean}} = \frac{10PR}{R + 9P}
$$

A chunk penalty is applied to penalize fragmented alignments, hypotheses
where the matched words are spread across many non-contiguous chunks rather
than appearing in a coherent sequence:

$$
\text{METEOR} = F_{\text{mean}} \cdot \left(1 - \frac{\text{chunks}}{\text{matched words}}\right)^3
$$

where "chunks" is the number of contiguous matched sequences and "matched words"
is the total count of matched words. A hypothesis that matches the reference in
one continuous block receives no penalty; one that matches in many small
fragments is penalized substantially.

### What METEOR Improves Over BLEU

METEOR's synonym and stemming matching partially addresses the paraphrase
problem: "automobile" matching "car" is better than BLEU's zero credit for
the same synonym. The recall weighting addresses BLEU's pure-precision focus.
The chunk penalty addresses BLEU's insensitivity to word order beyond n-gram
boundaries.

Empirically, METEOR shows better correlation with human judgments than BLEU
for machine translation, particularly at the sentence level where BLEU is most
unreliable. 

METEOR's limitations: the synonym coverage depends on WordNet, which is
English-centric, limited to certain word classes, and static (it does not
cover domain-specific terminology). METEOR has seen less adoption than BLEU
and ROUGE partly because WordNet availability varies across languages and
partly because the additional complexity did not translate into dramatically
better human correlation in downstream practice.

---

## Shared Failure Modes

All three metrics share the same fundamental limitations that make them
inadequate as the primary evaluation instrument for modern language models.

### The Synonymy Problem

None of the three metrics handle synonymy satisfactorily. BLEU and ROUGE 
ignores it entirely. METEOR captures it only within WordNet's coverage, which
is limited and English-specific. A hypothesis that uses different but equally
valid vocabulary than the reference receives a lower score than one that copies
the reference's phrasing, a systematic bias against natural language diversity.

This limitation was tolerable when machine translation produced outputs with
limited vocabulary variety. It is intolerable for general language model
evaluation where the hypothesis space is the full vocabulary in all combinations.

### The Single Valid Output Assumption

All three metrics assume that the references capture the space of acceptable
outputs. For tasks with genuinely diverse acceptable outputs (open-ended
question answering, instruction following, creative writing, explanation)
this assumption fails. A model that generates a high-quality response will
receive a low score if that response differs from the reference in arbitrary
ways.

The multiple-reference mitigation (computing metrics against several human-
written references and taking the maximum or average) partially addresses this
but requires expensive collection of multiple references and still undercovers
the space of acceptable outputs.

### Insensitivity to Factual Accuracy

N-gram overlap metrics are blind to factual content. A hypothesis that copies
the reference's sentence structure while substituting incorrect facts for
correct ones may receive nearly as high a score as the correct reference.
A fluent, well-organized, confidently stated wrong answer receives a higher
BLEU score than a correct answer expressed in different vocabulary.

This limitation is particularly dangerous for evaluating knowledge-intensive
generation: summaries, question answering, factual explanations. A model that
produces fluent but factually incorrect summaries will not be identified as
problematic by ROUGE evaluation.



---


## The Transition to Semantic Metrics

The inadequacy of n-gram overlap metrics for modern language model evaluation
motivated the development of semantic metrics that compare meaning rather
than lexical form. BERTScore, BLEURT, and their successors use
pretrained language model embeddings to compare hypothesis and reference in
a continuous semantic space, capturing synonym and paraphrase relationships
that n-gram metrics miss.

The transition from lexical to semantic metrics tracks the capability
improvement of the models being evaluated. When machine translation outputs
were close paraphrases of the reference (limited vocabulary, constrained
sentence structure), BLEU was a reasonable proxy. As translation models
improved and outputs became more diverse and fluent, the gap between BLEU
and human judgment widened. The same dynamic applies to summarization:
ROUGE was reasonable when abstractive models produced extractive-like
outputs; it became less appropriate as models produced genuinely abstractive
paraphrases.

For current language model evaluation (open-ended instruction following,
conversational responses, complex reasoning) neither n-gram metrics nor
embedding-based semantic metrics are adequate as primary evaluation instruments.
The limitations of all reference-based metrics in the open-ended generation
regime motivate the LLM-as-judge approach and, ultimately, the primacy of human 
evaluation.

---

## Key Takeaways

- BLEU measures modified n-gram precision (how much of the hypothesis appears
  in references) with a brevity penalty; it was designed for corpus-level
  machine translation evaluation with multiple references and is unreliable
  at the sentence level.
- ROUGE measures n-gram recall (how much of the reference appears in the
  hypothesis), with ROUGE-1, ROUGE-2, and ROUGE-L as standard variants; it
  was designed for summarization evaluation and favors extractive over
  abstractive approaches.
- METEOR extends BLEU with recall weighting, stemming, and WordNet synonymy
  matching, and adds a chunk penalty for fragmented alignments; it shows
  slightly better human correlation than BLEU but has seen less adoption due
  to WordNet's English-centricity and limited domain coverage.
- All three metrics share fundamental limitations: they penalize synonymy
  and paraphrase, assume the references cover the space of acceptable outputs,
  are insensitive to factual accuracy, and are noisy at the sentence level.
- A hypothesis using different but equally valid vocabulary than the reference
  receives a lower BLEU/ROUGE score than one that copies the reference's
  phrasing — a systematic bias against natural language diversity that becomes
  catastrophic for open-ended generation evaluation.
- The brevity penalty in BLEU prevents gaming by short hypotheses but does
  not prevent gaming by verbose hypotheses that copy many reference n-grams;
  there is no penalty for excessive length.
- N-gram metrics remain appropriate for corpus-level MT evaluation with
  multiple references, within-system regression testing, extractive
  summarization, and comparability with prior work.
- The correlation between BLEU and human judgments degrades as model outputs
  become more fluent and diverse; the metric was calibrated on 2002-era
  machine translation systems and the gap to human judgment has widened
  consistently as models improved.
- Factual incorrectness is invisible to n-gram overlap metrics: a fluent,
  confidently stated wrong answer receives nearly as high a score as a
  correct answer using different vocabulary — the primary motivation for
  hallucination metrics.
- The transition from lexical to semantic metrics (BERTScore, BLEURT)
  and from reference-based to reference-free metrics (LLM-as-judge)
  reflects the growing inadequacy of n-gram overlap for evaluating
  the open-ended generation tasks that define current language model deployment.

---

## Further Reading

- Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002). *BLEU: a Method
  for Automatic Evaluation of Machine Translation.* ACL. — The founding paper;
  the modified n-gram precision formulation and the brevity penalty are
  introduced here; the corpus-level correlation results with human judgments
  established BLEU as the MT evaluation standard for two decades.

- Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
  ACL Workshop. — Introduces ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S; the
  correlation analysis with human judgments on DUC summarization tasks is
  the key empirical contribution; the package implementation made ROUGE
  the de facto summarization evaluation standard.

- Banerjee, S., & Lavie, A. (2005). *METEOR: An Automatic Metric for MT
  Evaluation with Improved Correlation with Human Judgments.* ACL Workshop.
  — Introduces METEOR with stem and synonym matching; the human correlation
  improvements over BLEU at the sentence level are the key contribution.

- Callison-Burch, C., Osborne, M., & Koehn, P. (2006). *Re-evaluating the
  Role of BLEU in Machine Translation Research.* EACL. — The most influential
  critique of BLEU; demonstrates that BLEU does not correlate with human
  judgments in several important evaluation scenarios and that systems ranked
  differently by BLEU and humans exist in practice.

- Reiter, E. (2018). *A Structured Review of the Validity of BLEU.*
  Computational Linguistics. — Systematic review of 135 papers that used
  BLEU; documents the frequency of misuse (sentence-level evaluation,
  single reference, cross-system comparison without protocol matching)
  and the correlation with human judgments across studies.

- Mathur, N., Baldwin, T., & Cohn, T. (2020). *Tangled up in BLEU:
  Reevaluating the Evaluation of Automatic Machine Translation Evaluation
  Metrics.* ACL. — Meta-evaluation of MT metrics including BLEU, ROUGE,
  and METEOR; demonstrates that all three correlate poorly with direct
  assessment human judgments on modern MT systems; motivates the transition
  to embedding-based metrics.

- Deutsch, D., Dou, F., & Roth, D. (2022). *Examining the Premises and
  Implications of Reference-Based MT Evaluation.* NAACL. — Formal analysis
  of when reference-based evaluation is valid; the single-valid-output
  assumption is examined precisely and its violation quantified across
  translation tasks.

---
