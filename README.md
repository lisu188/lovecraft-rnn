# Lovecraft recurrent text model

A runnable, offline text continuator with numerical weights already initialized from 73 English prose texts and variants by H. P. Lovecraft. The implementation is one Python file; the initialized parameter arrays are supplied separately in `lovecraft_weights.npz`.

This is a **compiled sparse ReLU recurrent neural network implementing an interpolated n-gram language model**. It encodes the corpus's local word and punctuation statistics directly into weights. It has not undergone gradient training. Its supplied context is four tokens, so it cannot maintain long narrative plans, reliably answer lore questions, or guarantee consistency with the stories. Generated text can reproduce substantial original passages.

## Run immediately

Clone the repository and enter its directory:

```bash
git clone https://github.com/lisu188/lovecraft-rnn.git
cd lovecraft-rnn
```

Alternatively, download and extract the repository ZIP. Install NumPy:

```bash
python -m pip install -r requirements.txt
```

Continue your text:

```bash
python lovecraft_rnn.py generate --prompt "In the sunken city of R'lyeh," --tokens 250 --seed 7
```

Use an interactive prompt loop:

```bash
python lovecraft_rnn.py generate --interactive --tokens 250
```

View architecture and corpus information:

```bash
python lovecraft_rnn.py inspect
```

Verify the neural recurrence and serialization:

```bash
python lovecraft_rnn.py self-test
```

The program requires Python 3.10 or newer and NumPy. It was executed with Python 3.12 and NumPy 2.3.5. Inference needs only `lovecraft_rnn.py`, `lovecraft_weights.npz`, and NumPy. The corpus is included for inspection and rebuilding. Generation performs no network requests and does not read the corpus file.

## What is supplied

| File | Purpose |
| --- | --- |
| `lovecraft_rnn.py` | Complete model, compiler, sampler, CLI, and built-in verification |
| `lovecraft_weights.npz` | Initialized recurrent/input weights, biases, sparse output probabilities, vocabulary, and metadata |
| `lovecraft_corpus.json` | The 73 source texts, kept as separate documents |
| `corpus_manifest.json` | Source URLs, text hashes, scope, and excluded collection entries |
| `coverage_audit.json` | Title-by-title comparison with the Archive fiction index |
| `model_info.json` | Exact model dimensions, initialization method, corpus hash, and model hash |
| `generation_samples.json` | Three actual sampled outputs, with prompts, settings, and measured durations |
| `validation_report.json` | Independent numerical verification and the resolved temperature edge case |
| `runtime_metrics.json` | Measurements from this execution environment |
| `requirements.txt` | The only external dependency |
| `SHA256SUMS.txt` | Checksums of the supplied files |

## Actual initialized model

| Property | Value |
| --- | ---: |
| Corpus records | 73 |
| Whitespace-delimited corpus words | 477,889 |
| Corpus characters | 2,765,416 |
| Model tokens, including one end token per record | 549,233 |
| Vocabulary entries, including reserved entries | 27,275 |
| Recurrent neurons | 1,211,675 |
| Nonzero recurrent connections | 1,211,674 |
| Nonzero input connections | 1,211,674 |
| Nonzero empirical output weights | 1,753,212 |
| Maximum active neurons at one time | 5 |
| Context length | 4 word/punctuation/paragraph tokens |
| Default context scale | 4 |
| Gradient training steps | 0 |

The large logical hidden layer is sparse. Only the empty context and up to four matching suffix contexts are active at a time. The weight archive is approximately 8.6 MiB. One measured process used about 124.5 MiB of peak resident memory, loaded the model in about 0.32 seconds, and completed 140-token requests in about 0.55 seconds. These are local measurements, not guaranteed timings on another machine; load time is separate from generation time.

## How the weights are initialized

Each observed context `c` has a hidden neuron. Its parent is the same context without its final token. The empty context is a constant neuron equal to one.

For a nonempty context, the update is:

```text
h_c(t) = ReLU(h_prefix(c)(t - 1) + x_last(c)(t) - 1)
h_empty(t) = 1
```

Here `x` is the one-hot current input token. Therefore every nonempty context neuron has exactly one recurrent weight of `+1`, one input weight of `+1`, and a bias of `-1`. Its activation is exactly one when that context matches the input suffix. The root has zero input/recurrent weights and bias `+1`.

For every context and successor token, the compiler counts occurrences within each document and initializes the output coefficient to:

```text
P(v | c) = count(c followed by v) / sum_u count(c followed by u)
score(v) = sum_c h_c * context_scale ** len(c) * P(v | c)
p(v) = score(v) / sum_u score(u)
```

The unigram root provides a fallback for unseen contexts. Beginning and end tokens prevent the compiler from treating the end of one story and the beginning of another as adjacent prose.

The arrays `parent`, `input_token`, `recurrent_weight`, `input_weight`, and `bias` specify the initialized neural matrices sparsely. `output_indptr`, `output_token`, and `output_weight` store the output probabilities in sparse rows indexed by context; `context_weight` stores the diagonal context scaling layer. `depth`, `vocabulary`, and `metadata_json` complete the model.

Sparse execution uses the exact active-state equivalent of these fixed gates. The program also exposes `dense_matrices`, `dense_step`, and `dense_probabilities` for verifying small models. Materializing the full model's dense matrices would waste enormous amounts of memory and is intentionally prevented.

The relationship between recurrent networks and finite-state language models is established research. This implementation's correctness was checked numerically; the cited research does not imply that this model understands the stories:

- Anej Svete and Ryan Cotterell, *Recurrent Neural Language Models as Probabilistic Finite-state Automata*, EMNLP 2023: https://aclanthology.org/2023.emnlp-main.502/

## Corpus coverage

The package contains 62 texts extracted from a 67-text collection, plus 11 additional original texts and variants from the H. P. Lovecraft Archive. The five omitted collection entries are collaborations: *Medusa's Coil*, *Poetry of the Gods*, *The Crawling Chaos*, *The Horror at Martin's Beach*, and *Through the Gates of the Silver Key*.

The additional records include *History of the Necronomicon*, *Ibid*, *Old Bugs*, *Sweet Ermengarde*, *A Reminiscence of Dr. Samuel Johnson*, the extant juvenilia, both versions of *The Mysterious Ship*, and the surviving discarded-draft portions of *The Shadow over Innsmouth*.

Our title-level audit on 9 November 2026 matched all 71 linked, extant single-author entries selected from the Archive's alphabetical fiction index. Both *Mysterious Ship* versions are separate records, and *Imprisoned with the Pharaohs* (also titled *Under the Pyramids*, credited with Harry Houdini) is an additional record. This reconciles the 71 selected index entries with 73 corpus records. The full mapping, selection rule, and exclusions are in `coverage_audit.json`.

This is coverage of that selected index, not a claim to include every textual edition or all Lovecraft-related lore. Poetry, including *Fungi from Yuggoth*, essays, letters, most collaborations and revisions, later Mythos authors, adaptations, modern translations, and lost works are excluded. The discarded Innsmouth draft is fragmentary; one chart illustration in *The Little Glass Bottle* is omitted. The audit verifies title inclusion, not a line-by-line collation against manuscripts.

Sources:

- Collection: https://gutenberg.net.au/ebooks21/2100541h.html
- Collection source terms: https://gutenberg.net.au/licence.html
- Archive fiction index: https://www.hplovecraft.com/writings/fiction/
- Archive electronic texts: https://www.hplovecraft.com/writings/texts/
- *History of the Necronomicon*: https://www.hplovecraft.com/writings/texts/fiction/hn.aspx

Source URLs and SHA-256 hashes are recorded per document. The corpus uses original English prose. Publisher navigation, branding, and added introductory material are excluded. This package is an independent model experiment. Source public-domain determinations do not establish identical copyright status in every jurisdiction. Historical language and prejudices are retained in the source texts and may appear in generated output.

## Sampling controls

```bash
python lovecraft_rnn.py generate --prompt "Beneath the ruins of R'lyeh," --tokens 300 --temperature 1.0 --top-p 0.98 --context-scale 2 --seed 17
```

| Option | Behavior |
| --- | --- |
| `--tokens` | Maximum number of additional tokens; generation can end earlier at a document end token |
| `--temperature 0` | Greedy continuation; repeated phrases are possible |
| `--temperature 0.9` | Default probability sharpening |
| `--temperature 1.0` | Keeps the model's untempered probabilities before filtering |
| `--top-p 0.95` | Default nucleus probability cutoff |
| `--top-k 0` | No top-k limit; positive values cap candidate count |
| `--context-scale 4` | Stronger preference for the longest matching context |
| `--context-scale 2` | More contribution from shorter contexts, usually more recombination |
| `--seed` | Reproducible sampling for the same software, model, and arguments |
| `--keep-going` | Starts a new document state after sampling an end token |
| `--interactive` | Repeated prompt entry; an empty line exits |

Larger context scale and lower temperature usually favor copied phrases. Smaller context scale or higher temperature increases variation and can reduce grammatical coherence. These controls do not create semantic understanding.

Tokenization is case-sensitive. Unknown words clear unmatched context states while preserving the user's prompt in the output. An unknown final word that is a prefix of known vocabulary can be completed; a final token already in the vocabulary is treated as a complete token.

## Rebuild or add your own lore

Reproduce the supplied numerical model from the included corpus:

```bash
python lovecraft_rnn.py build lovecraft_corpus.json --model rebuilt_weights.npz --context 4 --context-scale 4
```

The reproducible network arrays and probabilities are deterministic. ZIP timestamps and added release metadata can make archive bytes differ from the supplied release, even when the numerical weights are identical.

Add a local UTF-8 text file while retaining the original corpus:

```bash
python lovecraft_rnn.py build lovecraft_corpus.json my_lore.txt --model expanded_weights.npz --context 4 --context-scale 4
```

Use the expanded model:

```bash
python lovecraft_rnn.py generate --model expanded_weights.npz --prompt "Beyond the ruined gates," --tokens 300
```

A directory input recursively reads its `.txt` files. JSON inputs may be arrays of objects with `text`, `title`, and optional `url` fields. Rebuilding with `--context 6` or another supported value increases the finite phrase window and typically increases memory and the tendency to reproduce source passages. It does not supply long-term narrative memory.

## Verification and limitations

The built-in self-test verifies 130 steps against explicit dense matrix multiplication, output equivalence, serialization, seeded generation, unknown prompts, and partial-word completion. A separate independent check covered 625 exhaustive short sequences, 2,500 exact recurrent-state comparisons, and 7,500 output-probability comparisons against independently counted n-grams. Invalid sampling settings were checked, and an extreme small-temperature numerical issue was fixed and verified. NPZ files contain numerical and Unicode arrays and are loaded with `allow_pickle=False`.

No held-out language-model quality score or factual lore benchmark is claimed. Sampling examples are actual outputs, not edited demonstrations of accuracy. The model's useful role is immediate, inspectable phrase-level text continuation and experimentation with direct weight initialization. For reliable lore answers or sustained original stories, it would need a substantially more capable model and an evidence-grounding mechanism.
