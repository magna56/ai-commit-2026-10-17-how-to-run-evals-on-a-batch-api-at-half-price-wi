# How to Run Evals on a Batch API at Half Price Without Dropping Cases From the Score

**TL;DR:** A batch eval comes back with some cases missing, and averaging what came back scores a different test set. Resubmit what failed in transit, count what the model rejected as a failure, and report coverage next to the score.

Published from [The AI Commit](https://theaicommit.com/#2026-10-17/code) — Evals & Reliability, 2026-10-17.

## Run

```bash
python3 code_example.py
```

## Output

```
custom_id for 'qa/long-017.v2': qa-long-017-v2-7e52740a

model-a
  first batch: 11 cases came back without an answer you can retry
  score over returned results:  79.9%   <- the number a naive harness prints
  coverage after resubmits:     100.0%
  score, a missing case = fail: 79.0%

model-b
  first batch: 12 cases came back without an answer you can retry
  score over returned results:  80.6%   <- the number a naive harness prints
  coverage after resubmits:     96.0%
  score, a missing case = fail: 77.2%
  16 failed for a reason tied to the case, e.g. qa/long-084.v2: rejected: prompt is too long

B minus A, score over returned results: +0.7 points
B minus A, every case counted:          -1.8 points
compare on the 384 cases both models answered, and report the rest as coverage

report() on a run with a pending case: 1 cases still pending, e.g. qa/x: resubmit first

```

## Code

See [`code_example.py`](code_example.py).
