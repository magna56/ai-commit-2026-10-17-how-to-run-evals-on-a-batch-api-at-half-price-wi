"""Run an eval suite on a batch API at half price, without dropping cases from the score.

A batch API (Anthropic Message Batches, OpenAI Batch) bills half price and gives you back
an unordered file in which some cases are missing. A harness that averages whatever came
back is scoring a different test set than last time. Every case needs exactly one outcome:
a verdict, a resubmit, or a failure with a reason.

Lift these into your harness: to_custom_id(), classify(), reconcile(), report().
The provider is faked so this runs with no API key; the result shapes match the docs.
Run: python3 code_example.py
"""
import hashlib, random, re

ID_OK = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # Anthropic's custom_id rule; OpenAI accepts any unique string


def to_custom_id(case_id):
    """Case ids like 'qa/long-017.v2' are not valid custom_ids. Map them, and keep the map."""
    if ID_OK.match(case_id):
        return case_id
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", case_id)[:50]
    return safe + "-" + hashlib.sha1(case_id.encode()).hexdigest()[:8]


def classify(line):
    """One result line -> ('answer', text) | ('resubmit', why) | ('fail', why).
    The rule: retry what went wrong in transit, fail what is a property of the case."""
    r = line["result"]
    if r["type"] == "succeeded":
        stop = r["message"]["stop_reason"]
        if stop == "end_turn":
            return "answer", r["message"]["content"][0]["text"]
        if stop == "pause_turn":          # server tools: the turn is unfinished, continue it
            return "resubmit", "pause_turn"
        return "fail", "stopped: " + stop  # max_tokens and friends: the answer is cut off
    if r["type"] in ("expired", "canceled"):
        return "resubmit", r["type"]       # never reached the model, and not billed
    err = r["error"]["type"]
    if err == "invalid_request_error":
        return "fail", "rejected: " + r["error"]["message"]  # same input fails every time
    return "resubmit", err                 # api_error, overloaded_error: transient


def reconcile(case_ids, lines, id_map):
    """Exactly one outcome per case, in case order. Anything absent is resubmitted."""
    back = {to: case for case, to in id_map.items()}
    got = {}
    for line in lines:                      # results arrive in any order
        case = back.get(line["custom_id"])
        if case is None or case in got:
            raise ValueError("unknown or duplicate custom_id: " + line["custom_id"])
        got[case] = classify(line)
    return {c: got.get(c, ("resubmit", "absent from results")) for c in case_ids}


def report(outcomes, verdicts):
    """Two numbers, never one. Refuse a score while cases are still in flight."""
    pending = [c for c, (kind, _) in outcomes.items() if kind == "resubmit"]
    if pending:
        raise RuntimeError(f"{len(pending)} cases still pending, e.g. {pending[0]}: resubmit first")
    n = len(outcomes)
    answered = [c for c, (kind, _) in outcomes.items() if kind == "answer"]
    passed = sum(verdicts[c] for c in answered)
    return {"coverage": len(answered) / n, "score": passed / n,
            "failed": {c: why for c, (kind, why) in outcomes.items() if kind == "fail"}}


# ---------- a fake provider, so the harness above has something to reconcile ----------
CASES = [f"qa/short-{i:03d}" for i in range(300)] + [f"qa/long-{i:03d}.v2" for i in range(100)]
TOKENS = {c: 4_000 if "short" in c else 120_000 + 3_000 * int(c[8:11]) for c in CASES}
MODELS = {"model-a": {"window": 1_000_000, "p": (0.85, 0.60)},
          "model-b": {"window": 370_000, "p": (0.88, 0.55)}}


def fake_batch(model, case_ids, id_map, expire_rate=0.03, seed=0):
    rng, m, lines = random.Random(f"{model}{seed}"), MODELS[model], []
    for case in case_ids:
        cid = id_map[case]
        if rng.random() < expire_rate:
            lines.append({"custom_id": cid, "result": {"type": "expired"}})
        elif TOKENS[case] > m["window"]:
            lines.append({"custom_id": cid, "result": {"type": "errored", "error": {
                "type": "invalid_request_error", "message": "prompt is too long"}}})
        else:
            right = random.Random(model + case).random() < m["p"]["long" in case]
            lines.append({"custom_id": cid, "result": {"type": "succeeded", "message": {
                "stop_reason": "end_turn", "content": [{"text": "right" if right else "wrong"}]}}})
    rng.shuffle(lines)                       # the docs: order is not guaranteed
    return lines


def naive_score(lines):
    """What a harness that reads only the successes reports. This is the bug."""
    ok = [l for l in lines if l["result"]["type"] == "succeeded"]
    return sum(l["result"]["message"]["content"][0]["text"] == "right" for l in ok) / len(ok)


def run(model):
    id_map = {c: to_custom_id(c) for c in CASES}
    first = fake_batch(model, CASES, id_map)
    outcomes = reconcile(CASES, first, id_map)
    retry = [c for c, (k, _) in outcomes.items() if k == "resubmit"]
    pending = retry
    for seed in range(1, 4):                 # resubmit until nothing is in flight, then stop
        if not pending:
            break
        outcomes.update(reconcile(pending, fake_batch(model, pending, id_map, seed=seed), id_map))
        pending = [c for c in pending if outcomes[c][0] == "resubmit"]
    verdicts = {c: why == "right" for c, (k, why) in outcomes.items() if k == "answer"}
    return naive_score(first), retry, report(outcomes, verdicts)


def main():
    print("custom_id for 'qa/long-017.v2':", to_custom_id("qa/long-017.v2"))
    rows = {}
    for model in MODELS:
        naive, retry, rep = run(model)
        rows[model] = (naive, rep)
        print(f"\n{model}")
        print(f"  first batch: {len(retry)} cases came back without an answer you can retry")
        print(f"  score over returned results:  {naive:.1%}   <- the number a naive harness prints")
        print(f"  coverage after resubmits:     {rep['coverage']:.1%}")
        print(f"  score, a missing case = fail: {rep['score']:.1%}")
        if rep["failed"]:
            case, why = next(iter(rep["failed"].items()))
            print(f"  {len(rep['failed'])} failed for a reason tied to the case, e.g. {case}: {why}")
    a, b = rows["model-a"], rows["model-b"]
    print(f"\nB minus A, score over returned results: {(b[0] - a[0]) * 100:+.1f} points")
    print(f"B minus A, every case counted:          {(b[1]['score'] - a[1]['score']) * 100:+.1f} points")
    shared = [c for c in CASES if c not in a[1]["failed"] and c not in b[1]["failed"]]
    print(f"compare on the {len(shared)} cases both models answered, and report the rest as coverage")
    try:
        report({"qa/x": ("resubmit", "expired")}, {})
    except RuntimeError as e:
        print("\nreport() on a run with a pending case:", e)


if __name__ == "__main__":
    main()
