#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PCTRAIN_ROOT = ROOT / "outputs/paper_5seed_qwen_counter_clean_pctrain_20260616"


def main() -> None:
    changed = 0
    checked = 0
    for path in sorted(PCTRAIN_ROOT.glob("*/seed_*/*/metrics.json")):
        checked += 1
        data = json.loads(path.read_text(encoding="utf-8"))
        args = data.get("args") or {}
        meta = data.setdefault("data", {})
        uses_evidence = bool(args.get("use_evidence_features") and (args.get("evidence") or args.get("llm_evidence")))
        updates = {
            "uses_evidence": uses_evidence,
            "evidence_mode": args.get("evidence_mode") if uses_evidence else "none",
            "evidence_path": args.get("evidence"),
            "llm_evidence_path": args.get("llm_evidence"),
        }
        if any(meta.get(k) != v for k, v in updates.items()):
            meta.update(updates)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            changed += 1
    print(f"checked={checked} changed={changed} root={PCTRAIN_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

