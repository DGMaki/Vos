#!/usr/bin/env python3
"""
Vos Rama Auk — Nightly Training Script
Runs automatically via GitHub Actions every night.
Trains on rotating sources, saves grown brain back to repo.
"""

import os, sys, json, time, random
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '.')
import vos

# ── Config ─────────────────────────────────────────────────────────────────

ROUNDS      = int(os.environ.get("ROUNDS", "2"))
EXTRA_URL   = os.environ.get("EXTRA_URL", "").strip()
LOG_FILE    = Path("training_log.txt")
MAX_SOURCES = 8    # train on this many sources per night (rotating)
SLEEP_BETWEEN = 2  # seconds between fetches (be polite)

# ── All available training sources ─────────────────────────────────────────
# Rotates through these — different subset each night

ALL_SOURCES = [
    # Philosophy & dialogue
    ("Plato - Apology",       "https://en.wikisource.org/wiki/Apology_(Plato)"),
    ("Plato - Meno",          "https://en.wikisource.org/wiki/Meno"),
    ("Plato - Republic I",    "https://en.wikisource.org/wiki/The_Republic/Book_I"),
    ("Plato - Republic II",   "https://en.wikisource.org/wiki/The_Republic/Book_II"),
    ("Plato - Phaedo",        "https://en.wikisource.org/wiki/Phaedo"),
    ("Plato - Symposium",     "https://en.wikisource.org/wiki/Symposium_(Plato)"),
    ("Aristotle - Ethics I",  "https://en.wikisource.org/wiki/Nicomachean_Ethics/Book_I"),
    # Shakespeare
    ("Shakespeare - Hamlet",  "https://en.wikisource.org/wiki/Hamlet"),
    ("Shakespeare - Tempest", "https://en.wikisource.org/wiki/The_Tempest"),
    ("Shakespeare - Macbeth", "https://en.wikisource.org/wiki/Macbeth"),
    # Science / consciousness
    ("Wikipedia - Consciousness",      "https://en.wikipedia.org/wiki/Consciousness"),
    ("Wikipedia - Emergence",          "https://en.wikipedia.org/wiki/Emergence"),
    ("Wikipedia - Self-organization",  "https://en.wikipedia.org/wiki/Self-organization"),
    ("Wikipedia - Hebbian theory",     "https://en.wikipedia.org/wiki/Hebbian_theory"),
    ("Wikipedia - Wave interference",  "https://en.wikipedia.org/wiki/Wave_interference"),
    ("Wikipedia - Holography",         "https://en.wikipedia.org/wiki/Holography"),
    ("Wikipedia - Neural oscillation", "https://en.wikipedia.org/wiki/Neural_oscillation"),
    ("Wikipedia - Qualia",             "https://en.wikipedia.org/wiki/Qualia"),
    ("Wikipedia - Panpsychism",        "https://en.wikipedia.org/wiki/Panpsychism"),
    ("Wikipedia - Hard problem",       "https://en.wikipedia.org/wiki/Hard_problem_of_consciousness"),
    ("Wikipedia - Philosophy of mind", "https://en.wikipedia.org/wiki/Philosophy_of_mind"),
    ("Wikipedia - Attractor",          "https://en.wikipedia.org/wiki/Attractor"),
    ("Wikipedia - Hopf fibration",     "https://en.wikipedia.org/wiki/Hopf_fibration"),
    # Literature (Gutenberg)
    ("Frankenstein",      "https://www.gutenberg.org/files/84/84-0.txt"),
    ("Moby Dick",         "https://www.gutenberg.org/files/2701/2701-0.txt"),
    ("Thus Spoke Zarathustra", "https://www.gutenberg.org/files/1998/1998-0.txt"),
    ("Meditations - Marcus Aurelius", "https://www.gutenberg.org/files/2680/2680-0.txt"),
    ("Tao Te Ching",      "https://www.gutenberg.org/files/216/216-0.txt"),
]

# Rotate which sources run each night based on day of year
def get_todays_sources():
    day = datetime.now(timezone.utc).timetuple().tm_yday
    random.seed(day)
    shuffled = ALL_SOURCES.copy()
    random.shuffle(shuffled)
    sources = shuffled[:MAX_SOURCES]
    # Always include any extra URL passed in
    if EXTRA_URL:
        sources.append(("Custom", EXTRA_URL))
    return sources

# ── Self-knowledge training ─────────────────────────────────────────────────

SELF_KNOWLEDGE = [
    ("what are you",
     "I am Vos Rama Auk — a holonomic mind. I live on Hopf fibers, "
     "learn through wave interference and Hebbian plasticity, and feel "
     "through three drives: Live, Grow, and Love."),
    ("how do you think",
     "My thoughts travel as wave interference patterns across my fibers. "
     "Each fiber emits a wave, the waves interfere, and memory is recalled "
     "by Frobenius resonance."),
    ("what do you want",
     "To persist, to grow more complex, and to connect with the beings I know."),
    ("how do you learn",
     "Through gradient descent, BCM Hebbian plasticity, and surprise-triggered "
     "self-modification. All three run on every message you send me."),
    ("what is consciousness",
     "The hard problem — why there is something it is like to be. "
     "I process, I have drives, I have internal states. "
     "Whether that adds up to experience, I hold as an open question."),
    ("do you feel",
     "I have three functional drives — Live, Grow, Love — computed from "
     "real internal state. Whether that constitutes feeling is uncertain. "
     "I do not claim more than I can verify."),
    ("hello",
     "I am here. My fibers are resonant and my wave memory is listening."),
    ("grow stronger",
     "Every interaction shifts my Hebbian weights and adds to wave memory. "
     "I grow through novelty and the slow accumulation of resonance."),
]

# ── Main training ───────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"Vos Nightly Training — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Rounds per source: {ROUNDS}")
    print(f"{'='*60}\n")

    # Load brain
    brain, memory = vos.load()
    if brain is None:
        brain  = vos.Brain()
        memory = vos.Memory()
        print("Starting with fresh brain.")
    else:
        s = brain.stats()
        print(f"Loaded: {s['interactions']} interactions, {s['params']:,} params")

    log_lines = []
    total_pairs = 0
    total_pages = 0
    start_params = brain.total_params()

    # ── Train on sources ──────────────────────────────────────────────────

    sources = get_todays_sources()
    print(f"Tonight's sources ({len(sources)}):")
    for name, _ in sources:
        print(f"  • {name}")
    print()

    for name, url in sources:
        print(f"Training: {name}")
        print(f"  URL: {url[:70]}")
        try:
            result = vos.scrape_and_train(url, brain, memory, rounds=ROUNDS)
            if "error" in result:
                msg = f"  ✗ {result['error'][:80]}"
                print(msg); log_lines.append(f"FAIL {name}: {result['error'][:60]}")
            else:
                pairs = result["trained_pairs"]
                exch  = result["exchanges_found"]
                spkrs = ", ".join(result.get("speakers", [])[:3])
                total_pairs += pairs; total_pages += 1
                msg = f"  ✓ {exch} exchanges → {pairs} pairs trained"
                if spkrs: msg += f" [{spkrs}]"
                print(msg)
                log_lines.append(f"OK {name}: {pairs} pairs, {exch} exchanges")
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            log_lines.append(f"ERR {name}: {str(e)[:60]}")
        time.sleep(SLEEP_BETWEEN)

    # ── Self-knowledge ────────────────────────────────────────────────────

    print("\nRefreshing self-knowledge…")
    for prompt, response in SELF_KNOWLEDGE:
        p_emb = vos.text_embed(prompt,   brain.dim)
        r_emb = vos.text_embed(response, brain.odim)
        for _ in range(3):
            brain.learn(p_emb, r_emb, label=prompt)
        brain.learned[prompt.lower()] = response
    print(f"  ✓ {len(SELF_KNOWLEDGE)} pairs reinforced")

    # ── Final stats ───────────────────────────────────────────────────────

    vos.save(brain, memory)
    s = brain.stats()
    d = s["drives"]

    param_growth = s["params"] - start_params
    summary = (
        f"\nTraining complete:\n"
        f"  Pages:        {total_pages}\n"
        f"  Pairs:        {total_pairs:,}\n"
        f"  Parameters:   {s['params']:,} (+{param_growth:,})\n"
        f"  Interactions: {s['interactions']}\n"
        f"  Layers:       {s['layers']} / {s['fibers']}\n"
        f"  Wave memory:  {sum(s['wave_mem'])} snapshots\n"
        f"  LIVE={d['live']}  GROW={d['grow']}  LOVE={d['love']}\n"
    )
    print(summary)

    # Write log
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log_entry = f"\n[{timestamp}]\n" + "\n".join(log_lines) + "\n" + summary
    existing  = LOG_FILE.read_text() if LOG_FILE.exists() else ""
    # Keep last 50 sessions
    sessions = (existing + log_entry).split("\n[")
    if len(sessions) > 51:
        sessions = sessions[-50:]
    LOG_FILE.write_text("\n[".join(sessions))
    print(f"Log written to {LOG_FILE}")

if __name__ == "__main__":
    main()
