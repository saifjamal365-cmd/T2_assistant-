"""Pick a small, fair slice of eval/dataset.json for a quick demo run.

Just taking the first 50 questions would be unfair - the 500 are grouped by
topic, so the first 50 would miss most departments and both languages might
not be represented evenly. This instead takes a proportional share from each
(language, family) group, so the 50-question set still looks like a small,
balanced version of the real 500 - same mix of English/Arabic, exact wording/
paraphrased/unanswerable.

Run it with: python scripts/build_eval_subset.py [N]   (default N=50)
Writes: eval/dataset_subset.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "eval" / "dataset.json"
OUT = ROOT / "eval" / "dataset_subset.json"
SEED = 42


def build_subset(size: int) -> list[dict]:
    items = json.loads(SOURCE.read_text(encoding="utf-8"))
    rng = random.Random(SEED)

    groups: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        key = (item["language"], item["family"])
        groups.setdefault(key, []).append(item)

    # spread each group's picks across departments instead of picking randomly,
    # so a small sample doesn't accidentally land on just one or two departments
    for group in groups.values():
        group.sort(key=lambda item: (item["department"] or "", item["id"]))

    share = size / len(items)
    counts = {key: round(len(group) * share) for key, group in groups.items()}

    # rounding can drift the total off `size` by a little - fix it up
    while sum(counts.values()) != size:
        key = rng.choice(list(counts))
        step = 1 if sum(counts.values()) < size else -1
        if 0 <= counts[key] + step <= len(groups[key]):
            counts[key] += step

    subset = []
    for key, group in groups.items():
        n = counts[key]
        step = max(1, len(group) // n) if n else 0
        picks = [group[i] for i in range(0, len(group), step)][:n]
        subset.extend(picks)

    rng.shuffle(subset)
    return subset


def main() -> None:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    subset = build_subset(size)

    OUT.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

    languages = {item["language"] for item in subset}
    print(f"Wrote {len(subset)} questions to {OUT}")
    print(f"  answerable: {sum(1 for i in subset if i['answerable'])}")
    print(f"  unanswerable: {sum(1 for i in subset if not i['answerable'])}")
    for lang in sorted(languages):
        print(f"  {lang}: {sum(1 for i in subset if i['language'] == lang)}")


if __name__ == "__main__":
    main()
