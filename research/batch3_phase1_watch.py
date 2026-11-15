"""Poll batch3_state.json; notify agent when a screen completes."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

STATE = Path(__file__).parent / "batch3_state.json"
POLL_SEC = 120
HEARTBEAT_EVERY = 5  # ticks ≈ 10 min at 120s

PROMPT = (
    "Batch 3 Phase 1 progress for Annabelle: read research/batch3_state.json, "
    "tail the active batch3_run.py terminal log for [SCREEN lines, "
    "summarize completed screens (val MAE, hard50_10), what's running now, "
    "and eta. If all primary screens done, note that they should run "
    "--skip-screening for promotions/report."
)


def snapshot() -> tuple[str, list[str], dict]:
    raw = STATE.read_text(encoding="utf-8") if STATE.exists() else "{}"
    h = hashlib.md5(raw.encode()).hexdigest()
    try:
        st = json.loads(raw)
    except json.JSONDecodeError:
        st = {}
    done = sorted(k for k, v in st.get("screening", {}).items() if v.get("done"))
    return h, done, st


def main() -> None:
    last_hash = ""
    last_done: list[str] = []
    ticks = 0
    while True:
        h, done, st = snapshot()
        ticks += 1
        changed = h != last_hash
        new_screens = [k for k in done if k not in last_done]
        heartbeat = ticks % HEARTBEAT_EVERY == 0

        if changed and new_screens:
            lines = ["New Phase 1 screen(s):"]
            for k in new_screens:
                sc = st["screening"][k]
                lines.append(
                    f"  {k} val={sc.get('val_insp_mae'):.4f} "
                    f"hard={sc.get('hard50_10_mae', 'NA')}"
                )
            detail = " | ".join(lines)
            print(
                f'AGENT_LOOP_WAKE_BATCH3 {{"prompt":"{PROMPT} {detail}"}}',
                flush=True,
            )
            last_hash = h
            last_done = done
        elif heartbeat:
            progress = f"{len(done)} done: {done}" if done else "no screens committed yet (training in progress)"
            print(
                f'AGENT_LOOP_WAKE_BATCH3 {{"prompt":"{PROMPT} Heartbeat — {progress}"}}',
                flush=True,
            )
            last_hash = h
            last_done = done

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)