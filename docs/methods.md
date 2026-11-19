# Source audit and declared adaptations

## DeepJIT

Reference: `baseline implementations/DeepJIT/model.py` and `padding.py` from
https://github.com/soarsmu/DeepJIT.
The model contains a message CNN, a changed-line CNN and a second CNN over line vectors,
then dropout and a binary sigmoid classifier. The architecture and initialization are loaded
unchanged from source. Defaults use its 64-dimensional embeddings, 64 filters, widths 1/2/3,
512 hidden units, dropout 0.5 and Adam learning rate 1e-4. The brief's epoch, early-stopping,
batch and validation policies replace the original training loop. Weight decay is explicit.

The input adapter uses an explicit regex lexer rather than undocumented external pickle
preprocessing. It preserves canonical case but lowercases neural tokens to match source
mapping behavior. Separate train-only code/message dictionaries have PAD=0 and UNK=1;
source code conflates unknowns with padding. Changed lines stay in file/hunk display order,
without addition/deletion prefixes, following the source model's single code channel.
Consequently T1 can be a CNN encoding no-op. This is recorded per commit. Head truncation
is explicit and audited; source defaults are 256 message tokens and 10 × 512 changed-code tokens.
Convolutions consume fixed padded arrays; as in the source, padding embeddings are learned
and no neural attention mask is added. Empty-message inputs are finite zero-ID sequences.

## CC2Vec

Reference: `jit_cc2ftr_model.py`, `jit_cc2ftr_train.py` and `jit_padding.py`
from https://github.com/CC2Vec/CC2Vec.
Reuse the reference bidirectional GRUs, word/line attention, file averaging, subtraction,
elementwise product, cosine similarity, Euclidean distance, standard comparison layer and
neural tensor comparisons. The resulting vector has `3 * embed_size + 4` components.
A separately fitted affine layer and sigmoid map that vector to defect probability.
Commit message text supplies auxiliary targets during encoder training.
The source's file slots (not a newly invented hunk GRU) are used; the declared but unused
hunk RNN remains unused. File order and within-file changed-line order are retained.

Correctness changes in a local subclass, without editing upstream files:

- Replace CUDA-only tensor creation and hidden states with the configured device.
- Transpose word/batch axes correctly and support variable final batches.
- Normalize attention along sequence positions; the source uses the batch dimension,
  causing an example's encoding to depend on unrelated batch members.
- Train on training records only; the original entry point concatenates training and test.
- Use all training batches; the original loop truncates its batch list to ten.
- Learn multi-hot **training message-term presence** with BCEWithLogitsLoss on raw logits;
  the original passes an already-sigmoid output to that loss. Padding/unknown IDs are
  excluded from the auxiliary objective, and validation-only terms do not extend dictionaries.
- Use the declared learning rate for Adam rather than the source's regularization variable.
- Select encoder and classifier checkpoints on the appropriate validation objectives.

The code encoder is frozen after auxiliary validation selection. Vectors are extracted without
gradients for downstream fitting and all prediction conditions; none are fitted on test data.
The encoder uses file/line/word input slots (defaults 2 × 10 × 64), with separate added and
removed channels. File contents are flattened across canonical hunks, matching the reference
file adapter; raw hunk boundaries remain available for controlled transformations.
Each fitted encoder computes its training and validation vectors once. These vectors
are cached in CPU memory for downstream classifier training and threshold selection.
Test and stress inference compute vectors from the inputs in each batch.

These fixes produce a documented CC2Vec pipeline adaptation, not a claim to exactly reproduce
historical pretrained outputs. Existing pretrained state dictionaries are not automatically
compatible with the new train-only vocabularies.

## JITLine–ApacheJIT adaptation

Reference: `JITLine/my_util.py` and replication notebooks from
https://github.com/awsm-research/JITLine-replication-package.
Use source literal normalization (`<STR>`, `<NUM>`), punctuation replacement, case preservation,
leading-`#` line skipping, per-channel duplicate-line removal and pooled added/deleted text.
Stable sorting replaces the original unordered set iteration. CountVectorizer retains its
default token pattern and source `min_df=3`, explicitly disables lowercasing, and fits only training code.
Count values are floating-point sparse values rather than overflowing dense int8 arrays.
The source notebook removes Python common tokens; this cross-language adaptation retains
them (`remove_python_common_tokens=False`) instead of applying a Python keyword list to Java.

The numeric feature mapping is exactly these ApacheJIT columns, in saved order:
`la, ld, nf, nd, ns, ent, ndev, age, nuc, aexp, arexp, asexp`.
No IDs, timestamps, project identity, `fix` flags or labels enter model features.
These do not reproduce all original JITLine metrics. Do not describe this as exact JITLine.
Fractional history metrics are preserved. Missing/invalid metrics exclude records from the
common cohort rather than being filled with invented numbers.

Random forest: 300 trees, Gini, bootstrap, no class weighting or depth limit,
minimum split=2, leaf=1, max_features=sqrt, explicit random_state and worker cap.
Numeric columns are standardized using training statistics before distance-based resampling;
token counts remain sparse and unscaled. SMOTE uses sparse input and at most minority_count−1
neighbors. Balanced training skips unnecessary SMOTE; an insufficient minority skips a
synthetic-sampling search and records that adaptation.

Bounded differential evolution selects neighbor count on
the first seed's validation ROC-AUC, with fixed RF settings. Initial population=10,
mutation=0.7 and source recombination=0.3 (with the mandatory mutant coordinate),
at most two additional generations, at most 30 actual candidate fits. SMOTE balances
to ratio=1.0; neighbor search is 1…20, constrained by the available minority count. Small
smoke populations below four do not perform DE mutation. Selected settings are reused across
all final seeds; forests and SMOTE are refit stochastically per seed. This bounded one-parameter
search budget is a compute-motivated adaptation of the original optimization. Predictions are raw
positive-class probabilities, without churn division or line localization.

## Data scope and transformation qualifications

The default source policy selects these textual source suffixes case-sensitively:
`.java .groovy .scala .kt .kts .py .c .h .cpp .hpp .js .ts .sh`.
Binary, rename-only and other non-code files are excluded from model text and counts of
filtered files are retained. Renames are rendered as deletion/addition using `--no-renames`.
No generated-file detector is assumed; generated source within the allowed suffixes is retained.
This explicit policy must be reported because the historical metrics can describe broader scope.

T1 exchanges every retained changed-line role. In each two-sided hunk, T2 exchanges old/new
blocks in each contiguous edit region, preserving channel order and intervening context;
empty region sides are exchanged too, but wholly one-sided hunks remain untouched. This
retains the full hunk's old/new channel contents and is involutive for normal Git diff order.
Context is not a
predictive input. T3 is a synthetic prefix to the first retained block in each adapter, inserted
once, with fixed vocabularies; it can become unknown tokens or displace truncated code.
T4 empties DeepJIT's message input. It is structurally inapplicable to CC2Vec and JITLine
because both predict from inputs that exclude the message. JITLine reversal tests are unavailable when
filtered file scope or exact la/ld counts disagree; metrics are never silently reconciled.
The retained numeric history and coverage fields otherwise remain fixed.

Cross-partition near-duplicate analysis, metric formula/unit verification against the dataset
construction code, historical label-availability dates and a hardware timing pilot still require
research review before publication. The executable pipeline records actual counts and compute;
it does not supply fabricated final cohort sizes or performance estimates.
