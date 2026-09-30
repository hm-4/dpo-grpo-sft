# Dataset used by HM.ipynb

The notebook uses **GSM8K math word problems with an assignment-specific split**
for training a language model to solve arithmetic problems. These are the six
files linked in the notebook's `initialize_project_data()` function. They were
downloaded and inspected on 2026-09-29. All files are JSONL: one JSON object per
line.

| Stage | Training file and rows | Evaluation file and rows | Fields in each raw row |
| --- | --- | --- | --- |
| Supervised fine-tuning (SFT) | `data/train.jsonl`: 8,692 | `data/eval.jsonl`: 100 | `question`, `answer` |
| Direct preference optimization (DPO) | `data/dpo_train.jsonl`: 1,000 | `data/dpo_eval.jsonl`: 100 | `question`, `chosen`, `rejected` |
| Group relative policy optimization (GRPO) | `data/grpo_train.jsonl`: 1,000 | `data/grpo_eval.jsonl`: 100 | `question`, `reasoning`, `answer` |

## How the notebook uses them

- **SFT:** `answer` contains a worked solution, followed by `####` and the final
  number. `transform_gsm8k()` separates this into `reasoning` and `answer`. The
  model learns to generate the solution and answer from the question. Prompt
  tokens are excluded from the training loss.
- **DPO:** each question has a preferred response (`chosen`) and a dispreferred
  response (`rejected`). The model learns to favor the chosen response relative
  to a frozen reference model. A rejected response can have faulty reasoning
  even when its final number happens to match the chosen response.
- **GRPO:** the file also includes `reasoning`, but `GRPODataset` uses the
  question and final answer. Training samples several responses per question
  and rewards a matching extracted final answer. The completed implementation
  scores only generated response text and ignores thousands separators in
  numeric answers. It normalizes rewards within each prompt group and updates
  the policy using a token-level GRPO loss with frozen-reference regularization.

Evaluation rows are held out from parameter updates to measure performance.
There are **8,792 distinct question strings across all six files**, rather than
10,992 independent problems: DPO and GRPO use the first 1,000 SFT training
questions, and all three stages share the same 100 evaluation questions in the
same order.

## Verified provenance and changed split

The SFT question and answer strings were compared against the raw
[official OpenAI GSM8K training file](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl)
and
[official test file](https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl),
whose paths are documented in the
[OpenAI dataset repository](https://github.com/openai/grade-school-math).
The combined assignment SFT pool has exactly the same 8,792 question-to-answer
mappings as the combined official pool: every question and answer string
matches exactly. The split differs:

| Assignment file | Questions from official train (7,473) | Questions from official test (1,319) |
| --- | ---: | ---: |
| `data/train.jsonl` | 7,473 | 1,219 |
| `data/eval.jsonl` | 0 | 100 |

Therefore this assignment trains on 1,219 questions from the standard GSM8K test
set. Its own 100 evaluation questions are held out from its training data, but
its results should not be reported as scores on the standard GSM8K test split.
These findings establish SFT content identity; they do not establish how the
assignment's DPO rejected responses were generated.

Reference files used for this comparison are cached locally in
`.cache/gsm8k-provenance/`; the assignment data was not changed.

## Example from the files

The first training problem asks how many clips Natalia sold if she sold 48 in
April and half as many in May. The correct solution computes `48 / 2 = 24`, then
`48 + 24 = 72`.

In SFT, this solution and `#### 72` are together in `answer`. In GRPO,
`reasoning` contains the solution and `answer` is the string `"72"`. DPO pairs
the correct solution with a rejected solution containing incorrect arithmetic.
In this particular pair both responses end in `72`, illustrating why preference
data is about the whole response, not just the final number.

## Validation of the downloaded files

- Every row parses as JSON, has the expected keys, and contains nonempty string
  values.
- Each SFT answer contains exactly one `####` delimiter, as required by the
  notebook's parser.
- Each DPO chosen response contains one `####`. Rejected responses without that
  delimiter occur in 63 training rows and 8 evaluation rows; the DPO loader does
  not require a delimiter in rejected responses.
- No repeated question appears within any one file. There is no train/evaluation
  question overlap within SFT, DPO, or GRPO, using both exact string comparison
  and comparison after collapsing whitespace and ignoring case. This is a text
  overlap check, not a proof that no two problems are semantically similar.
- DPO contains 23 training pairs and 2 evaluation pairs whose text after `####`
  matches between chosen and rejected responses. No chosen/rejected pair is
  completely identical. GRPO answers match the corresponding DPO chosen final
  answers.

## Sources specified in the notebook

| File | Google Drive source |
| --- | --- |
| `train.jsonl` | [Download](https://drive.google.com/uc?id=1M1DEfy6CVPx8HU4abzT71zyrUfePRXw0) |
| `eval.jsonl` | [Download](https://drive.google.com/uc?id=1jjWJ3x7AjLK5Q35LIQAheRO32d-KZz3D) |
| `dpo_train.jsonl` | [Download](https://drive.google.com/uc?id=16vEtdw_fO_3TyVzVg0uey-HH2teujEsw) |
| `dpo_eval.jsonl` | [Download](https://drive.google.com/uc?id=1cxUpymIRHux4oFUt3qBK9lFj8bQc4-8u) |
| `grpo_train.jsonl` | [Download](https://drive.google.com/uc?id=1YSdeXppWrkFeKZwKLWTdKAPoj4huhBjb) |
| `grpo_eval.jsonl` | [Download](https://drive.google.com/uc?id=10WzbpEVRHtxzniLwSgoyIlytk-BrgWol) |

Launch the notebook with `HM` as the working directory so its relative
`./data/...` paths resolve. These downloaded files are local data, not new
repository publications.
