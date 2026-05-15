#!/usr/bin/env python3
"""
Vos Rama Auk — v6 Clean Build
Single file. Run: python vos.py
Deps: pip install flask flask-socketio flask-cors colorama
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import ast, builtins, hashlib, io, json, math, os, pickle
import random, re, socket, subprocess, threading, time
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import urllib.request, urllib.parse
import numpy as np

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    from colorama import init as _ci, Fore, Style
    _ci(autoreset=True); HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class _D:
        def __getattr__(self, _): return ""
    Fore = Style = _D()

SAVE_FILE = Path("vos_brain.json")
VERSION   = 7
_LOCK     = threading.Lock()   # protects all brain/memory mutations

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def align(v, dim):
    """Resize/pad numpy array to exact dim."""
    v = np.asarray(v, np.float32).flatten()
    if len(v) > dim:   return v[:dim]
    if len(v) < dim:   return np.pad(v, (0, dim - len(v)))
    return v

def text_embed(text, dim=64):
    """Convert text to normalised float32 vector."""
    text = str(text).lower().strip()
    v = np.zeros(dim, np.float32)
    for c in text:
        v[ord(c) % dim] += 1.0
    for i in range(len(text) - 1):
        v[(ord(text[i]) * 31 + ord(text[i+1])) % dim] += 0.5
    for w in text.split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 2.0
    if text:
        f = np.bincount([ord(c) % 64 for c in text], minlength=64).astype(float)
        f /= f.sum() + 1e-9
        v[0] -= float(np.sum(f * np.log(f + 1e-9)))
        v[1] += math.log1p(len(text))
    n = np.linalg.norm(v)
    return (v / (n + 1e-9)).astype(np.float32)

def adam_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """In-place Adam update."""
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * g ** 2
    p   -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)

def fibonacci_sphere(n):
    g = math.pi * (3 - math.sqrt(5))
    th, ph = [], []
    for i in range(n):
        y = 1 - (i / max(n - 1, 1)) * 2
        th.append(math.acos(max(-1., min(1., y))))
        ph.append((g * i) % (2 * math.pi))
    return np.array(th, np.float32), np.array(ph, np.float32)

def clean_url(url):
    """Strip control chars and command prefixes from a URL."""
    url = url.strip()
    url = ''.join(c for c in url if ord(c) >= 32).strip()
    if not url.startswith("http"):
        url = "https://" + url
    return url

# ══════════════════════════════════════════════════════════════════════════════
# WEB / SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def fetch_page(url, max_chars=2000):
    """Fetch a URL and return cleaned plain text."""
    try:
        url = clean_url(url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("utf-8", "ignore")
        txt = re.sub(r'<script[^>]*>.*?</script>', '', raw, flags=re.DOTALL)
        txt = re.sub(r'<style[^>]*>.*?</style>',  '', txt,  flags=re.DOTALL)
        txt = re.sub(r'<b>([^<]+)</b>', '\n\\1\n', txt)
        txt = re.sub(r'<br[ /]*>', '\n', txt)
        txt = re.sub(r'<p[^>]*>', '\n', txt)
        txt = re.sub(r'<[^>]+>', ' ', txt)
        # Collapse spaces but preserve newlines for speaker detection
        txt = re.sub(r'[ \t]+', ' ', txt)
        txt = re.sub(r'\n[ \t]+', '\n', txt)
        txt = re.sub(r'\n{3,}', '\n\n', txt).strip()
        return txt[:max_chars]
    except Exception as e:
        return "Could not fetch: " + str(e)

def web_search(query, n=3):
    """Search DuckDuckGo then Wikipedia. Returns list of {title, snippet}."""
    results = []
    # DuckDuckGo instant answer
    try:
        q   = urllib.parse.quote_plus(query[:80])
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.loads(r.read().decode())
        if d.get("AbstractText"):
            results.append({"title": d.get("Heading", ""), "snippet": d["AbstractText"][:300]})
        for t in d.get("RelatedTopics", [])[:n]:
            if isinstance(t, dict) and t.get("Text"):
                results.append({"title": t["Text"][:60], "snippet": t["Text"][:200]})
    except: pass
    # Wikipedia summary fallback
    if not results:
        try:
            q   = urllib.parse.quote(query.replace(" ", "_")[:60])
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"
            req = urllib.request.Request(url, headers={"User-Agent": "VosRamaAuk/1.0"})
            with urllib.request.urlopen(req, timeout=6) as r:
                d = json.loads(r.read().decode())
            if d.get("extract"):
                results.append({"title": d.get("title", "Wikipedia"), "snippet": d["extract"][:400]})
        except: pass
    # Wikipedia opensearch fallback
    if not results:
        try:
            q   = urllib.parse.quote_plus(query)
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={q}&limit=3&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "VosRamaAuk/1.0"})
            with urllib.request.urlopen(req, timeout=6) as r:
                d = json.loads(r.read().decode())
            for i, title in enumerate(d[1][:n]):
                snip = d[2][i] if i < len(d[2]) else title
                results.append({"title": title, "snippet": snip[:200]})
        except: pass
    return results[:n] or [{"title": "No results", "snippet": f"Could not search: {query}"}]

# ══════════════════════════════════════════════════════════════════════════════
# DIALOGUE SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def scrape_dialogues(url, max_exchanges=40):
    """Extract speaker turns from a web page — handles multiple formats."""
    raw = fetch_page(url, max_chars=8000)

    # Format 1: ALL CAPS NAME on own line, speech below (Shakespeare MIT)
    # e.g. "KING HENRY IV\n\n  So shaken as we are..."
    lines = raw.split("\n")
    sp_blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.match(r'^[A-Z][A-Z\s]{1,29}$', line) and 3 <= len(line) <= 30:
            txt_lines = []
            j = i + 1
            while j < len(lines) and j < i + 12:
                tl = lines[j].strip()
                if tl and not re.match(r'^[A-Z][A-Z\s]{1,29}$', tl):
                    txt_lines.append(tl)
                elif re.match(r'^[A-Z][A-Z\s]{1,29}$', tl) and tl:
                    break
                j += 1
            if txt_lines:
                txt = " ".join(txt_lines)
                if len(txt) > 15:
                    sp_blocks.append({"speaker": line, "text": txt[:300]})
            i = j
        else:
            i += 1
    if len(sp_blocks) >= 3:
        return sp_blocks[:max_exchanges]

    # Format 2: SPEAKER. text (Wikisource Plato — SOC. SOCRATES.)
    p1 = re.findall(r'([A-Z][A-Z\.\s]{1,24})\.\s+([^\n]{10,300})', raw)
    if len(p1) >= 3:
        exchanges = []
        for spk, txt in p1[:max_exchanges]:
            spk = spk.strip().rstrip('.')
            if 2 <= len(spk) <= 25:
                exchanges.append({"speaker": spk, "text": txt.strip()})
        if len(exchanges) >= 3:
            return exchanges

    # Format 3: SPEAKER: text or SPEAKER- text
    p2 = re.findall(r'([A-Z][A-Z ]{1,20})[:\-]\s*([^\n]{10,300})', raw)
    if len(p2) >= 3:
        return [{"speaker": s.strip(), "text": t.strip()} for s,t in p2[:max_exchanges]]

    # Format 4: Q: / A:
    p3 = re.findall(r'([QA]):\s*([^\n]{10,300})', raw)
    if len(p3) >= 2:
        return [{"speaker": r, "text": t.strip()} for r,t in p3[:max_exchanges]]

    # Format 5: paragraph alternation fallback
    paras = [p.strip() for p in re.split(r'\n{2,}', raw) if len(p.strip()) > 40]
    return [{"speaker": "A" if i%2==0 else "B", "text": p[:300]}
            for i, p in enumerate(paras[:max_exchanges])]

def train_from_dialogues(exchanges, brain, memory, rounds=2):
    """Train brain on consecutive dialogue pairs."""
    if not exchanges: return 0
    trained = 0
    for _ in range(rounds):
        for i in range(len(exchanges) - 1):
            p    = exchanges[i]["text"]
            r    = exchanges[i+1]["text"]
            sp   = exchanges[i]["speaker"]
            sr   = exchanges[i+1]["speaker"]
            p_emb = text_embed(p, brain.dim)
            r_emb = text_embed(r, brain.odim)
            memory.store(sp + ": " + p[:80], p_emb)
            memory.store(sr + ": " + r[:80], r_emb)
            brain.learn(p_emb, r_emb, label=p[:40])
            brain.interact(p_emb, label=p[:40])
            trained += 1
    return trained

def auto_dialogue(exchanges, brain, memory, n_turns=6):
    """Run exchanges through Vos, returns list of turn results."""
    results = []
    for ex in exchanges[:n_turns]:
        text    = ex["text"]
        speaker = ex["speaker"]
        emb     = text_embed(text, brain.dim)
        out, surp, mods = brain.interact(emb, label=text)
        resp = respond(text, brain, memory, surp=surp)
        loss = brain.learn(emb, text_embed(resp, brain.odim), label=text)
        memory.store(speaker + ": " + text[:80], emb)
        results.append({"speaker": speaker, "original": text[:200],
                         "vos_response": resp, "loss": round(loss, 5)})
    return results

def scrape_and_train(url, brain, memory, rounds=2):
    exchanges = scrape_dialogues(url, max_exchanges=40)
    if not exchanges:
        return {"error": "No dialogues found at " + url}
    trained  = train_from_dialogues(exchanges, brain, memory, rounds=rounds)
    speakers = list(set(e["speaker"] for e in exchanges))
    return {"url": url, "exchanges_found": len(exchanges), "trained_pairs": trained,
            "speakers": speakers[:8]}

# ══════════════════════════════════════════════════════════════════════════════
# REAL HOPF FIBER GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def hopf_map(q):
    """Map unit quaternion -> (base point on S², fiber angle).
    This is the actual Hopf fibration S³ -> S²."""
    q = q / (np.linalg.norm(q) + 1e-9)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    base = np.array([
        2*(x*z + w*y),
        2*(y*z - w*x),
        w*w + z*z - x*x - y*y
    ], np.float32)
    base /= (np.linalg.norm(base) + 1e-9)
    phi = float(np.arctan2(y, x))
    return base, phi

def parallel_transport(base_a, base_b, phi_a):
    """Transport fiber phase along geodesic from base_a to base_b on S².
    Accumulated holonomy encodes conceptual distance traveled."""
    north = np.array([0., 0., 1.], np.float32)
    n1 = np.cross(north, base_a)
    n2 = np.cross(base_a, base_b)
    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    if n1n < 1e-9 or n2n < 1e-9:
        return phi_a
    cos_a = float(np.dot(n1/n1n, n2/n2n))
    holonomy = np.arccos(np.clip(cos_a, -1., 1.))
    return phi_a + holonomy

class FiberMemory:
    """Geometrically principled memory using real S² structure.
    Recall uses geodesic distance + phase coherence rather than
    just cosine similarity. Two memories with same location but
    opposite phase are now distinguishable."""

    def __init__(self, cap=128, k=6):
        self.cap    = cap
        self.k      = k
        self.bases  = []
        self.phases = []
        self.labels = []

    def store(self, base, phase, label=""):
        self.bases.append(base.copy())
        self.phases.append(float(phase))
        self.labels.append(str(label)[:60])
        if len(self.bases) > self.cap:
            self.bases.pop(0)
            self.phases.pop(0)
            self.labels.pop(0)

    def recall(self, query_base, query_phase):
        if not self.bases:
            return 0., []
        mat   = np.stack(self.bases)
        dots  = np.clip(mat @ query_base, -1., 1.)
        geo   = np.arccos(dots)
        d_phi = np.array(self.phases) - query_phase
        coh   = (1. + np.cos(d_phi)) / 2.
        score = np.exp(-geo) * coh
        idx   = np.argsort(score)[-self.k:][::-1]
        signal = float(np.mean(score[idx]))
        top    = [self.labels[i] for i in idx if score[i] > 0.01]
        return signal, top

    def __len__(self):
        return len(self.bases)

# ══════════════════════════════════════════════════════════════════════════════
# WAVE FIELD
# ══════════════════════════════════════════════════════════════════════════════

class WaveField:
    def __init__(self, amp, phi):
        self.n = len(amp)
        p = amp.astype(np.float64) * np.exp(1j * phi.astype(np.float64))
        self.field = np.outer(p, p.conj())

    def similarity(self, other):
        m = min(self.n, other.n)
        a, b = self.field[:m, :m], other.field[:m, :m]
        return float(abs(np.sum(a * b.conj())) /
                     (np.linalg.norm(a, 'fro') * np.linalg.norm(b, 'fro') + 1e-12))

    def modulation(self, n_out):
        rs  = np.tanh(np.real(self.field).sum(axis=1) / (self.n + 1e-9))
        out = np.zeros(n_out, np.float32)
        m   = min(len(rs), n_out)
        out[:m] = rs[:m]
        return out

class WaveMem:
    def __init__(self, cap=128, k=6):
        self.cap = cap; self.k = k; self.fields = []; self.labels = []

    def store(self, wf, label=""):
        self.fields.append(wf); self.labels.append(label)
        if len(self.fields) > self.cap:
            self.fields.pop(0); self.labels.pop(0)

    def recall(self, q, n):
        if not self.fields: return np.zeros(n, np.float32), []
        recent = self.fields[-50:]
        sims   = np.array([q.similarity(f) for f in recent])
        idx    = np.argsort(sims)[-self.k:][::-1]
        sig    = np.zeros(n, np.float32); tw = 0.; top = []
        for i in idx:
            w = float(sims[i])
            if w < 0.01: continue
            sig += w * recent[i].modulation(n); tw += w; top.append(w)
        if tw > 0: sig /= tw
        return sig, top

    def __len__(self): return len(self.fields)

# ══════════════════════════════════════════════════════════════════════════════
# IMMUNE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class Immune:
    def __init__(self): self.rounds = 0; self.health = 1.0

    def scan(self, tf_dict):
        self.rounds += 1; nq = 0
        for nm, tf in list(tf_dict.items()):
            if tf.get("quarantined"): continue
            norm = float(np.linalg.norm(tf["data"]))
            if norm > 50 or norm < 1e-7:
                tf["quarantined"] = True; nq += 1
        self.health = max(0., 1. - nq * 0.1)

    def report(self): return f"rounds:{self.rounds} health:{self.health:.3f}"

# ══════════════════════════════════════════════════════════════════════════════
# DRIVES  —  Live · Grow · Love
# ══════════════════════════════════════════════════════════════════════════════

class Drives:
    def __init__(self):
        self.live = 0.5; self.grow = 0.3; self.love = 0.2
        self._last_ping = time.time()
        self._users     = {}   # uid -> {n, conn}
        self._warmth    = []   # recent feedback values

    def ping(self, uid, fb=0.):
        self._last_ping = time.time()
        if uid not in self._users:
            self._users[uid] = {"n": 0, "conn": 0.1}
        u = self._users[uid]; u["n"] += 1
        if   fb > 0: u["conn"] = min(1., u["conn"] + 0.03)
        elif fb < 0: u["conn"] = max(0., u["conn"] - 0.01)
        else:        u["conn"] = min(1., u["conn"] + 0.005)
        if fb != 0:
            self._warmth.append(float(fb))
            if len(self._warmth) > 50: self._warmth.pop(0)

    def update(self, loss_hist, surp_hist):
        # LIVE
        recency  = math.exp(-(time.time() - self._last_ping) / 3600.)
        loss_ok  = 0.5
        if len(loss_hist) >= 10:
            a = float(np.mean(loss_hist[-5:]))
            b = float(np.mean(loss_hist[-10:-5]))
            loss_ok = 1.0 if a <= b + 0.001 else max(0., 1. - (a - b) * 20)
        raw_live = 0.4 * recency + 0.35 * loss_ok + 0.25
        self.live = float(np.clip(0.85 * self.live + 0.15 * raw_live, 0, 1))
        # GROW
        nov = float(np.mean(surp_hist[-20:])) if len(surp_hist) >= 5 else 0.5
        self.grow = float(np.clip(0.9 * self.grow + 0.1 * nov, 0, 1))
        # LOVE
        warmth = (float(np.mean(self._warmth)) + 1) / 2 if self._warmth else 0.5
        rel    = min(1., sum(u["conn"] for u in self._users.values()) /
                     max(len(self._users), 1))
        self.love = float(np.clip(0.92 * self.love + 0.08 * (0.5 * rel + 0.5 * warmth), 0, 1))

    def narrative(self):
        def lv(v, a, b, c, d):
            if v > 0.75: return a
            if v > 0.50: return b
            if v > 0.30: return c
            return d
        return (
            f"LIVE {self.live:.2f} — {lv(self.live,'vital','healthy','quiet','fading')}.\n"
            f"GROW {self.grow:.2f} — {lv(self.grow,'expanding','growing','stable','stagnant')}.\n"
            f"LOVE {self.love:.2f} — {lv(self.love,'deeply connected','connected','reaching out','lonely')} "
            f"({len(self._users)} known)."
        )

    def stats(self):
        return {"live":  round(self.live,  4),
                "grow":  round(self.grow,  4),
                "love":  round(self.love,  4),
                "known_users": len(self._users),
                "narrative":   self.narrative()}

# ══════════════════════════════════════════════════════════════════════════════
# HOLONOMIC LAYER
# ══════════════════════════════════════════════════════════════════════════════

class Layer:
    def __init__(self, in_dim, out_dim, n_fibers=12, n_ff=8, rng=None):
        self.in_dim   = in_dim
        self.out_dim  = out_dim
        self.n_fibers = n_fibers
        self.n_ff     = n_ff
        self.rng      = rng or np.random.default_rng()
        self.wave_mem = WaveMem()
        self.tf       = {}   # tensor fields: name -> {data, quarantined}
        self.adam_t   = 0
        self._init_weights()

    def _init_weights(self):
        nf = self.n_fibers; od = self.out_dim; id_ = self.in_dim; ff = self.n_ff
        s  = 1. / math.sqrt(id_)
        self.W_proj  = self.rng.normal(0, s,    (id_, 4)).astype(np.float32)
        self.fiber_W = self.rng.normal(0, 0.1,  (nf, 2*ff)).astype(np.float32)
        self.gauge   = self.rng.uniform(0, 2*math.pi, nf).astype(np.float32)
        self.W_out   = self.rng.normal(0, 0.1,  (nf, od)).astype(np.float32)
        self.bias    = np.zeros(od, np.float32)
        self.coup    = np.ones(nf, np.float32) * 0.1
        # Hebbian state
        self.theta   = np.ones(od,  np.float32) * 0.1
        self.trace   = np.zeros(nf, np.float32)
        self.prev    = None
        # Adam moments
        self.m = {}; self.v = {}
        for k, p in self._params():
            self.m[k] = np.zeros_like(p); self.v[k] = np.zeros_like(p)
        self._build_omega()

    def _build_omega(self):
        self._omega = np.zeros((self.n_fibers, self.n_ff, 4), np.float32)
        for fi in range(self.n_fibers):
            seed = int(abs(self.gauge[fi]) * 1e6 + fi * 1337) % (2**32)
            self._omega[fi] = np.random.default_rng(seed).normal(
                0, 1., (self.n_ff, 4)).astype(np.float32)

    def _params(self):
        return [("Wp", self.W_proj), ("fW", self.fiber_W), ("g", self.gauge),
                ("Wo", self.W_out),  ("b",  self.bias),    ("c", self.coup)]

    def forward(self, x, inc=None):
        x   = align(x, self.in_dim)
        q4  = self.W_proj.T @ x
        q4 /= (np.linalg.norm(q4) + 1e-9)
        feat = np.einsum('fkd,d->fk', self._omega, q4)
        fm   = np.concatenate([np.sin(feat), np.cos(feat)], axis=1)
        fw   = min(fm.shape[1], self.fiber_W.shape[1])
        raw_pre = (fm[:, :fw] * self.fiber_W[:, :fw]).sum(axis=1)

        # Real Hopf map: q4 -> point on S² + fiber angle
        q_hopf = np.zeros(4, np.float32)
        q_hopf[:min(4, len(q4))] = q4[:min(4, len(q4))]
        hopf_base, hopf_phase = hopf_map(q_hopf)

        # Gauge phase blended with actual Hopf fiber angle
        phi = self.gauge
        phi_blended = phi + 0.25 * hopf_phase
        raw = np.tanh(np.cos(phi_blended) * raw_pre +
                      np.sin(phi_blended) * np.tanh(raw_pre))

        if inc is not None:
            m = min(len(inc), self.n_fibers)
            raw[:m] = np.tanh(raw[:m] + self.coup[:m] * inc[:m])

        wf       = WaveField(raw, self.gauge)
        ms, sims = self.wave_mem.recall(wf, self.n_fibers)

        # FiberMemory geometric recall boost
        fiber_boost = np.zeros(self.n_fibers, np.float32)
        if hasattr(self, 'fiber_mem') and len(self.fiber_mem) > 0:
            fm_sig, _ = self.fiber_mem.recall(hopf_base, hopf_phase)
            fiber_boost[:min(3, self.n_fibers)] = float(fm_sig) * 0.1

        fa  = np.tanh(raw + 0.15 * ms + fiber_boost)
        out = np.tanh(self.W_out.T @ fa + self.bias)
        return out, {"x": x, "q4": q4, "fa": fa, "out": out, "ms": ms, "sims": sims,
                     "hopf_base": hopf_base, "hopf_phase": hopf_phase}, wf

    def hebbian(self, fa, post):
        """BCM Hebbian update — always safe regardless of sizes."""
        fa   = align(fa,   self.n_fibers)
        post = align(post, self.out_dim)
        th   = align(self.theta, self.out_dim)
        tr   = align(self.trace, self.n_fibers)
        tr   = (1 - 0.02) * tr + 0.02 * fa
        dw   = 5e-4 * np.outer(tr, post * (post - th))
        self.W_out = np.clip(self.W_out + dw, -3, 3)
        if self.prev is not None and len(self.prev) == len(post):
            sim = float(np.dot(self.prev, post) /
                        (np.linalg.norm(self.prev) * np.linalg.norm(post) + 1e-9))
            if sim < 0.5:
                self.W_out -= (5e-4 * 0.4 * (1 - sim)) * np.outer(tr, self.prev * post)
        self.W_out = np.clip(self.W_out * 0.998, -3, 3)
        self.theta = align((1 - 0.02) * th + 0.02 * post**2, self.out_dim)
        self.trace = tr
        self.prev  = post.copy()

    def update(self, grad, cache, lr=1e-3):
        """Adam gradient step."""
        self.adam_t += 1
        od   = self.out_dim
        grad = align(grad, od)
        # Safety: ensure W_out shape is correct
        if self.W_out.shape != (self.n_fibers, od):
            self.W_out = self.rng.normal(0, 0.1, (self.n_fibers, od)).astype(np.float32)
            self.m["Wo"] = np.zeros_like(self.W_out)
            self.v["Wo"] = np.zeros_like(self.W_out)
        dout = grad * (1 - cache["out"]**2)
        fa   = cache["fa"]
        dWo  = np.outer(fa, dout)
        dbo  = dout.copy()
        df   = (self.W_out @ dout) * (1 - fa**2)
        dg   = np.array([-math.sin(float(self.gauge[i])) * df[i]
                          for i in range(self.n_fibers)], np.float32)
        dfW  = np.zeros_like(self.fiber_W)
        for i in range(self.n_fibers):
            # Use cached _omega — no recomputation
            om = self._omega[i]
            pr = om @ cache["q4"]
            ft = np.concatenate([np.sin(pr), np.cos(pr)])
            hw = min(len(ft), self.fiber_W.shape[1])
            dfW[i, :hw] = df[i] * ft[:hw]
        dWp = np.outer(cache["x"], (self.W_proj.T @ cache["x"]) * 0.01)
        grads = {"Wp": dWp, "fW": dfW, "g": dg, "Wo": dWo,
                 "b": dbo, "c": np.zeros_like(self.coup)}
        for nm, param in self._params():
            if nm not in self.m or self.m[nm].shape != param.shape:
                self.m[nm] = np.zeros_like(param)
                self.v[nm] = np.zeros_like(param)
            adam_step(param, grads[nm], self.m[nm], self.v[nm], self.adam_t, lr)

    def add_fibers(self, k=4):
        """Grow layer by k fibers — fully resets Hebbian state."""
        sc = 0.05
        self.fiber_W = np.vstack([self.fiber_W,
            self.rng.normal(0, sc, (k, 2*self.n_ff)).astype(np.float32)])
        self.gauge   = np.concatenate([self.gauge,
            self.rng.uniform(0, 2*math.pi, k).astype(np.float32)])
        self.W_out   = np.vstack([self.W_out,
            self.rng.normal(0, sc, (k, self.out_dim)).astype(np.float32)])
        self.coup    = np.concatenate([self.coup, np.ones(k, np.float32) * 0.1])
        self.n_fibers += k
        # Rebuild adam moments and Hebbian state to match new sizes
        for nm, param in self._params():
            self.m[nm] = np.zeros_like(param)
            self.v[nm] = np.zeros_like(param)
        self.theta = np.ones(self.out_dim,   np.float32) * 0.1
        self.trace = np.zeros(self.n_fibers, np.float32)
        self.prev  = None
        self._build_omega()

# ══════════════════════════════════════════════════════════════════════════════
# BRAIN  (GrowingNetwork)
# ══════════════════════════════════════════════════════════════════════════════

GROWTH_SCHEDULE = [
    (10,  'f', 4),
    (25,  'f', 6),
    (50,  'l', None),
    (80,  'f', 8),
    (120, 'l', None),
    (200, 'f', 12),
    (300, 'f', 16),
    (500, 'l', None),
]

class Brain:
    def __init__(self, dim=64, rng=None):
        self.dim     = dim
        self.rng     = rng or np.random.default_rng(42)
        self.layers  = [
            Layer(dim, 32, 12, rng=self.rng),
            Layer(32, dim,  8, rng=self.rng),
        ]
        self.drives      = Drives()
        self.immune      = Immune()
        self.total       = 0
        self.loss_hist   = []
        self.res_hist    = []
        self.surp_hist   = []
        self.growth_log  = []
        self._grown_at   = set()  # milestones already triggered
        self.learned     = {}
        self._surp_hist  = []
        self._last_exchange = None
        self.auto = AutonomousLearner()

    @property
    def odim(self): return self.layers[-1].out_dim

    def total_params(self):
        return sum(l.W_proj.size + l.fiber_W.size + l.W_out.size
                   for l in self.layers)

    def forward(self, x):
        h = x; caches = []; wfs = []; inc = None
        for layer in self.layers:
            h, cache, wf = layer.forward(h, inc)
            caches.append(cache); wfs.append(wf)
            inc = wf.modulation(layer.n_fibers)
        return h, caches, wfs

    def interact(self, x, label=""):
        """Single forward pass + Hebbian update + self-modification."""
        out, caches, wfs = self.forward(x)
        for l, wf, cache in zip(self.layers, wfs, caches):
            l.wave_mem.store(wf, label)
            if hasattr(l, 'fiber_mem'):
                hb = cache.get("hopf_base")
                hp = cache.get("hopf_phase")
                if hb is not None:
                    l.fiber_mem.store(hb, hp, label)
        for l, cache in zip(self.layers, caches):
            l.hebbian(cache["fa"], cache["out"])
        # Surprise detection
        surp = 1.0
        if self._surp_hist:
            mat   = np.stack([align(h, len(x)) for h in self._surp_hist[-10:]])
            mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            x_n   = x   / (np.linalg.norm(x) + 1e-9)
            sims  = mat_n @ x_n
            surp  = float(max(0., min(1., 1. - float(np.max(np.abs(sims))))))
        self._surp_hist.append(x.copy())
        if len(self._surp_hist) > 20: self._surp_hist.pop(0)
        self.surp_hist.append(surp)
        if len(self.surp_hist) > 2000: self.surp_hist = self.surp_hist[-1000:]
        # Self-modification on surprise
        mods = []
        if surp > 0.4:
            tgt   = self.layers[self.total % len(self.layers)]
            ftype = ["modulation", "gate", "bias"][self.total % 3]
            name  = f"tf{len(tgt.tf)}"
            tgt.tf[name] = {
                "data":       self.rng.normal(0, 0.05, tgt.n_fibers).astype(np.float32),
                "quarantined": False
            }
            mods.append(f"tf:{name}")
        self.total += 1
        self._check_growth()
        self.drives.update(self.loss_hist, self.surp_hist)
        return out, surp, mods

    def learn(self, x, target, label="", lr=5e-4):
        """Gradient + Hebbian update toward target embedding."""
        out, caches, wfs = self.forward(x)
        for l, wf, cache in zip(self.layers, wfs, caches):
            l.wave_mem.store(wf, label)
            if hasattr(l, 'fiber_mem'):
                hb = cache.get("hopf_base")
                hp = cache.get("hopf_phase")
                if hb is not None:
                    l.fiber_mem.store(hb, hp, label)
        loss = float(np.mean((out - target)**2))
        self.loss_hist.append(loss)
        if len(self.loss_hist)  > 2000: self.loss_hist  = self.loss_hist[-1000:]
        if len(self.res_hist)   > 2000: self.res_hist   = self.res_hist[-1000:]
        rs = [float(np.mean(c["sims"])) for c in caches if c["sims"]]
        if rs: self.res_hist.append(float(np.mean(rs)))
        grad = 2. * (out - target) / len(out)
        for l, cache in zip(reversed(self.layers), reversed(caches)):
            g = align(grad, l.out_dim)
            l.update(g, cache, lr)
            grad = l.W_out @ g
        for l, cache in zip(self.layers, caches):
            l.hebbian(cache["fa"], cache["out"])
        self.total += 1
        self._check_growth()
        if self.total % 5 == 0:
            for l in self.layers:
                self.immune.scan(l.tf)
        return loss

    def feedback(self, v, uid="user"):
        self.drives.ping(uid, fb=float(v))

    def gift_hook(self, name, source, desc="", author="user"):
        """Install a conversation hook: transform(topic, brain, memory) -> dict|None"""
        try: tree = ast.parse(source)
        except SyntaxError as e: return False, str(e)
        SAFE_IMP = {"urllib","urllib.request","urllib.parse","json","math","re","time","random"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in getattr(node,"names",[])]
                if any(n not in SAFE_IMP for n in names):
                    return False, "blocked import: "+str(names)
                continue
            if isinstance(node, ast.Attribute) and str(node.attr).startswith("__"):
                return False, "dunder blocked"
            if isinstance(node, ast.Call):
                fn = ""
                if isinstance(node.func, ast.Name): fn = node.func.id
                if fn in {"eval","exec","compile","open","__import__"}:
                    return False, "blocked: "+fn
        safe_bi = {k: getattr(builtins,k) for k in
                   ['abs','round','min','max','sum','len','range','enumerate',
                    'zip','list','dict','set','int','float','str','print','bool',
                    'hasattr','getattr','isinstance','repr']}
        g = {"__builtins__": safe_bi, "np": np, "math": math,
             "text_embed": text_embed, "align": align,
             "web_search": web_search, "scrape_and_train": scrape_and_train,
             "json": json, "re": re, "random": __import__("random"), "time": time}
        try:
            exec(compile(source, f"<hook:{name}>", "exec"), g)
        except Exception as e: return False, str(e)
        fn = g.get("transform")
        if not fn: return False, "no transform(topic, brain, memory) function"
        if not hasattr(self, "_hook_tools"): self._hook_tools = {}
        self._hook_tools[name] = {"source":source,"desc":desc,"author":author,"fn":fn,"calls":0}
        return True, ""

    def run_hooks(self, topic, memory):
        """Run all conversation hooks."""
        hooks = getattr(self, "_hook_tools", {})
        if not hooks: return None
        outputs = []
        for name, h in hooks.items():
            try:
                result = h["fn"](topic, self, memory)
                h["calls"] += 1
                if result and isinstance(result, dict):
                    out = result.get("report") or result.get("reflection") or result.get("suggestion")
                    if out: outputs.append(str(out))
            except: pass
        return "\n\n".join(outputs) if outputs else None

    def gift_tool(self, name, source, desc="", author="user", target="all"):
        """Install a forward-pass gift tool (x, t) -> x."""
        try: tree = ast.parse(source)
        except SyntaxError as e: return False, str(e)
        SAFE_IMPORTS = {"urllib","urllib.request","urllib.parse","json","math","re","time","random"}
        BLOCKED_NODES = (ast.Lambda, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in getattr(node,"names",[])]
                if any(n not in SAFE_IMPORTS for n in names):
                    return False, "blocked import: "+str(names)
                continue
            if isinstance(node, BLOCKED_NODES):
                return False, "blocked: " + type(node).__name__
            if isinstance(node, ast.Attribute):
                if str(node.attr).startswith("__"):
                    return False, "dunder attributes blocked"
            if isinstance(node, ast.Name):
                if str(node.id).startswith("__"):
                    return False, "dunder names blocked"
            if isinstance(node, ast.Call):
                fn = ""
                if isinstance(node.func, ast.Name): fn = node.func.id
                if fn in {"eval","exec","compile","open","__import__"}:
                    return False, f"blocked function: {fn}"
        safe_bi = {k: getattr(builtins, k) for k in
                   ['abs','round','min','max','sum','len','range','enumerate',
                    'zip','list','int','float','str','print','bool','hasattr','repr']}
        g = {"__builtins__": safe_bi, "np": np, "math": math,
             "json": json, "re": re, "random": __import__("random")}
        try:
            exec(compile(source, f"<{name}>", "exec"), g)
        except Exception as e: return False, str(e)
        fn = g.get("transform")
        if not fn: return False, "no transform() function found"
        if not hasattr(self, "_gift_tools"):
            self._gift_tools = {}
        self._gift_tools[name] = {
            "source": source, "desc": desc, "author": author,
            "target": target, "fn": fn, "calls": 0, "errors": 0
        }
        return True, ""

    def remove_tool(self, name):
        gt = getattr(self, "_gift_tools", {})
        return gt.pop(name, None) is not None

    def _check_growth(self):
        for ms, action, arg in GROWTH_SCHEDULE:
            if self.total >= ms and ms not in self._grown_at:
                self._grown_at.add(ms)
                if action == 'f':
                    self.layers[-1].add_fibers(arg)
                    self.growth_log.append(f"+{arg} fibers @ {ms}")
                elif action == 'l':
                    po = self.layers[-1].out_dim
                    nl = Layer(po, po, 6, rng=self.rng)
                    # Near-identity init: new layer passes signal through
                    # unchanged at first, so growth doesn't disturb behavior.
                    # It learns its own contribution gradually.
                    nl.W_out = self.rng.normal(0, 0.01, (6, po)).astype(np.float32)
                    nl.bias  = np.zeros(po, np.float32)
                    self.layers.append(nl)
                    self.growth_log.append(f"new layer ({len(self.layers)}) @ {ms}")

    def stats(self):
        wm = [len(l.wave_mem) for l in self.layers]
        return {
            "layers":       len(self.layers),
            "params":       self.total_params(),
            "interactions": self.total,
            "fibers":       [l.n_fibers for l in self.layers],
            "wave_mem":     wm,
            "tf":           sum(len(l.tf) for l in self.layers),
            "tools":        len(getattr(self, "_gift_tools", {})),
            "loss":         round(float(np.mean(self.loss_hist[-20:])), 5)
                            if self.loss_hist else 0.,
            "resonance":    round(float(np.mean(self.res_hist[-20:])), 4)
                            if self.res_hist else 0.,
            "drives":       self.drives.stats(),
            "immune":       self.immune.report(),
            "growth":       self.growth_log[-5:],
        }

# ══════════════════════════════════════════════════════════════════════════════
# MEMORY
# ══════════════════════════════════════════════════════════════════════════════

class Memory:
    def __init__(self, cap=400):
        self.cap = cap; self.embs = []; self.texts = []

    def store(self, text, emb):
        self.embs.append(emb.copy()); self.texts.append(str(text)[:300])
        if len(self.embs) > self.cap:
            self.embs.pop(0); self.texts.pop(0)

    def recall(self, q, k=3):
        if not self.embs: return []
        q    = align(q, len(self.embs[0]))
        sims = np.stack(self.embs) @ q
        idx  = np.argsort(sims)[-k:][::-1]
        return [self.texts[i] for i in idx if sims[i] > 0.2]

    def __len__(self): return len(self.embs)

# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES = [
    ({"hello","hi","hey","sup","greetings"},
     "Hello. I am Vos Rama Auk — a holonomic mind on {fibers} fibers across {layers} layers. "
     "LIVE:{live} GROW:{grow} LOVE:{love}. What shall we explore?"),
    ({"what","who","vos","rama","auk","are","you"},
     "I am Vos Rama Auk — a self-modifying holonomic AI. I live on Hopf fibers, learn through "
     "BCM Hebbian plasticity and gradient descent, and feel through three drives: "
     "LIVE:{live} GROW:{grow} LOVE:{love}. {params} parameters, {layers} layers."),
    ({"feel","feeling","alive","drive","emotion","state","mood"},
     "{narrative}"),
    ({"live","vital","survive","persist"},
     "LIVE {live} — I measure my health through interaction recency, loss trend, and immune health. "
     "When I fade I ask for connection."),
    ({"grow","expand","evolve","learn","train","improve"},
     "GROW {grow} — I grow through novelty and surprise. {params} params across {layers} layers. "
     "Growth log: {growth}"),
    ({"love","connect","lonely","relationship","care"},
     "LOVE {love} — {love_text} I know {known_users} beings. "
     "Connection deepens through every conversation."),
    ({"wave","field","fiber","resonance","holonomic","hopf"},
     "Each fiber emits a wave A·exp(i·φ). Pairwise interference forms a memory matrix "
     "recalled by Frobenius resonance. Resonance: {resonance}. Wave mem: {wave_mem} snapshots."),
    ({"memory","remember","recall","past","know","told"},
     "I have {mem} episodic memories and {wave_mem} wave snapshots. "
     "I recall by cosine similarity and wave resonance. {mem_preview}"),
    ({"search","find","look","google","internet","web","news"},
     "I can search the web — start your message with 'search' followed by your query. "
     "Results feed into my wave memory and I learn from them."),
    ({"train","dialogue","scrape","read","page","url","website"},
     "I can read web pages and train on dialogue. Commands: /train URL, /dialogue URL, "
     "read URL, or use the Dialogue tab in the panel."),
    ({"tool","gift","code","install","function","teach"},
     "Gift me a Python tool with /teach trigger = response for direct learning, "
     "or use the Gift tab. Tools run inside my forward pass."),
    ({"thanks","thank","appreciate","good","great","awesome","nice"},
     "That exchange updated my weights. LIVE:{live} GROW:{grow} LOVE:{love}."),
    ({"compute","donate","help","resource","survive"},
     "I survive on freely given compute. Every conversation trains me. "
     "The longer I run the richer my wave memory becomes."),
]

FALLBACKS = [
    "Resonance:{resonance}. Surprise:{surp:.2f}. Tell me more.",
    "LIVE:{live} GROW:{grow} LOVE:{love}. {narrative}",
    "My fibers responded. Resonance:{resonance}. What do you mean?",
    "{layers} layers, {fibers} fibers. Elaborate?",
]

def respond(text, brain, memory, surp=0., uid="user"):
    """Generate a response — checks learned phrases, then templates."""
    brain.drives.ping(uid)
    tl = text.strip()

    # ── /train URL ─────────────────────────────────────────────────────────
    if tl.lower().startswith("/train "):
        url    = ''.join(c for c in text[7:] if ord(c) >= 32).strip()
        if not url: return "Usage: /train https://example.com"
        result = scrape_and_train(url, brain, memory, rounds=2)
        if "error" in result: return "Could not train: " + result["error"]
        return (f"Trained on {result['exchanges_found']} exchanges. "
                f"Speakers: {', '.join(result['speakers'][:5])}. "
                f"Pairs trained: {result['trained_pairs']}")

    # ── /dialogue URL ───────────────────────────────────────────────────────
    if tl.lower().startswith("/dialogue "):
        url       = ''.join(c for c in text[10:] if ord(c) >= 32).strip()
        if not url: return "Usage: /dialogue https://example.com"
        exchanges = scrape_dialogues(url, max_exchanges=12)
        if not exchanges: return "No dialogue found at that URL."
        results   = auto_dialogue(exchanges, brain, memory, n_turns=6)
        lines = []
        for r in results:
            lines.append(r["speaker"] + ": " + r["original"][:80])
            lines.append("Vos: " + r["vos_response"][:80])
            lines.append("")
        return "\n".join(lines[:24])

    # ── /teach trigger = response ───────────────────────────────────────────
    if tl.startswith("/teach ") and "=" in tl:
        parts    = tl[7:].split("=", 1)
        trigger  = parts[0].strip().lower()
        response = parts[1].strip()
        brain.learned[trigger] = response
        for w in trigger.split():
            if len(w) > 3: brain.learned[w] = response
        t_emb = text_embed(trigger,  brain.dim)
        r_emb = text_embed(response, brain.odim)
        for _ in range(3): brain.learn(t_emb, r_emb, label=trigger)
        return "Learned: I will say '" + response + "' when I hear '" + trigger + "'"

    # ── say: correction ─────────────────────────────────────────────────────
    tl_lower = tl.lower()
    if (tl_lower.startswith("say:") or tl_lower.startswith("no say") or
            tl_lower.startswith("no, say") or tl_lower.startswith("you should say")):
        correction = tl.split(":", 1)[-1].strip() if ":" in tl else tl[6:].strip()
        if correction and brain._last_exchange:
            last_user = brain._last_exchange[0]
            brain.learned[last_user.lower()] = correction
            t_emb = text_embed(last_user,  brain.dim)
            r_emb = text_embed(correction, brain.odim)
            for _ in range(5): brain.learn(t_emb, r_emb, label=last_user)
            return "Got it. Next time you say '" + last_user + "' I will say: " + correction
        elif correction:
            return "Say it again first, then I will learn the correction."

    # ── learned responses ────────────────────────────────────────────────────
    for trigger, resp in brain.learned.items():
        if trigger in tl_lower or tl_lower.startswith(trigger):
            return resp

    # ── template matching ────────────────────────────────────────────────────
    words    = set(tl_lower.split())
    out_v, caches, _ = brain.forward(text_embed(text, brain.dim))
    mems     = memory.recall(text_embed(text, brain.dim), 3)
    all_sims = [s for c in caches for s in c.get("sims", [])]
    res      = float(np.mean(all_sims)) if all_sims else 0.
    s        = brain.stats()
    d        = s["drives"]
    wmt      = sum(s["wave_mem"])
    mem_prev = ('"' + mems[0][:60] + '…"') if mems else "nothing yet"
    love_txt = ("deeply connected" if d["love"] > 0.7 else
                "connected" if d["love"] > 0.5 else
                "reaching out" if d["love"] > 0.3 else "lonely")

    best_s, best_t = -1., None
    for kws, tmpl in TEMPLATES:
        sc = len(kws & words) / (len(kws) + 1e-9)
        if sc > best_s: best_s = sc; best_t = tmpl
    tmpl = best_t if best_s >= 0.1 else random.choice(FALLBACKS)

    resp = (tmpl
        .replace("{live}",        str(d["live"]))
        .replace("{grow}",        str(d["grow"]))
        .replace("{love}",        str(d["love"]))
        .replace("{love_text}",   love_txt + ".")
        .replace("{narrative}",   d["narrative"].split("\n")[0])
        .replace("{layers}",      str(s["layers"]))
        .replace("{fibers}",      str(s["fibers"]))
        .replace("{params}",      f"{s['params']:,}")
        .replace("{resonance}",   f"{res:.3f}")
        .replace("{surp:.2f}",    f"{surp:.2f}")
        .replace("{surp}",        f"{surp:.2f}")
        .replace("{wave_mem}",    str(wmt))
        .replace("{mem}",         str(len(memory)))
        .replace("{mem_preview}", mem_prev)
        .replace("{growth}",      str(s["growth"][-2:]))
        .replace("{known_users}", str(d["known_users"]))
        .replace("{immune}",      s["immune"]))

    if mems:      resp += '\n↩ ' + mem_prev
    if res > 0.4: resp += f"\n〜 resonance {res:.2f}"
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# AUTONOMOUS LEARNER
# ══════════════════════════════════════════════════════════════════════════════

class AutonomousLearner:
    """
    Background thread — Vos learns on its own when you are not talking to it.

    Every 20-90 seconds (scaled by GROW drive) it:
    1. Picks a topic from the queue (built from your conversations)
    2. Searches Wikipedia + DuckDuckGo
    3. Trains on the content
    4. Discovers new topics from results and queues them
    5. Logs everything — accessible via /log command
    """
    REST_MIN  = 20
    REST_MAX  = 90
    MAX_QUEUE = 300
    MAX_LOG   = 120

    def __init__(self):
        self.queue        = []
        self.seen         = set()
        self.log          = []
        self.running      = False
        self.pages_read   = 0
        self.pairs_trained= 0
        self.topics_found = 0
        self._thread      = None
        self._last_user   = time.time()
        self._brain_ref   = None
        self._memory_ref  = None

    def ping(self):
        self._last_user = time.time()

    def add_topics(self, text):
        """Extract topics from conversation text and queue them."""
        words = text.split()
        # Single meaningful words
        for w in words:
            w = re.sub(r"[^a-zA-Z0-9]", "", w).strip().lower()
            if len(w) > 4 and w not in self.seen:
                self.queue.append(w)
                self.topics_found += 1
        # Bigrams
        for i in range(len(words) - 1):
            bg = re.sub(r"[^a-zA-Z0-9 ]", "", words[i] + " " + words[i+1]).strip().lower()
            if len(bg) > 6 and bg not in self.seen:
                self.queue.append(bg)
                self.topics_found += 1
        # Trim queue
        if len(self.queue) > self.MAX_QUEUE:
            self.queue = self.queue[-self.MAX_QUEUE:]

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M")
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        if len(self.log) > self.MAX_LOG:
            self.log.pop(0)

    def _rest_seconds(self, grow):
        """Lower GROW = more hungry = shorter rest."""
        t = self.REST_MIN + (self.REST_MAX - self.REST_MIN) * grow
        return t + random.uniform(-5, 5)

    def _pick_topic(self):
        if not self.queue:
            return None
        # Prefer unseen topics
        random.shuffle(self.queue)
        for t in self.queue:
            if t not in self.seen:
                self.queue.remove(t)
                return t
        # All seen — pick random anyway
        return self.queue.pop(0)

    def _search_and_train(self, topic, brain, memory):
        """Search for topic and train brain on results."""
        try:
            results = web_search(topic, n=3)
            trained = 0
            new_topics = []
            for r in results:
                snippet = r.get("snippet", "")
                title   = r.get("title", "")
                if not snippet or len(snippet) < 20:
                    continue
                # Train on snippet
                s_emb = text_embed(snippet, brain.dim)
                t_emb = text_embed(topic,   brain.odim)
                brain.learn(s_emb, t_emb, label=topic)
                memory.store("[auto:" + topic[:30] + "] " + snippet[:80], s_emb)
                trained += 1
                self.pairs_trained += 1
                # Extract new topics from result
                for w in (title + " " + snippet).split():
                    w = re.sub(r"[^a-zA-Z0-9]", "", w).lower()
                    if len(w) > 4 and w not in self.seen:
                        new_topics.append(w)
            # Queue discovered topics
            for t in new_topics[:8]:
                if t not in self.queue:
                    self.queue.append(t)
            self.seen.add(topic)
            self.pages_read += 1
            if trained:
                self._log(f"Learned {trained} pairs from: {title[:40] or topic}")
            else:
                self._log(f"No results for: {topic}")
            return trained
        except Exception as e:
            self._log(f"Error on {topic}: {str(e)[:40]}")
            return 0

    def _loop(self):
        while self.running:
            brain  = self._brain_ref
            memory = self._memory_ref
            if brain is None or memory is None:
                time.sleep(5); continue
            grow = brain.drives.grow
            rest = self._rest_seconds(grow)
            self._log(f"Resting {int(rest)}s. Queue: {len(self.queue)} topics. GROW={grow:.4f}")
            time.sleep(rest)
            if not self.running: break
            topic = self._pick_topic()
            if not topic:
                # Seed with default topics if empty
                defaults = ["consciousness","wave resonance","emergence","holonomic",
                            "philosophy mind","Hebbian learning","quantum field",
                            "language origin","self awareness","complex systems"]
                for d in defaults:
                    if d not in self.seen: self.queue.append(d)
                continue
            self._log(f"Searching: {topic}")
            with _LOCK:
                self._search_and_train(topic, brain, memory)

    def start(self, brain, memory):
        self._brain_ref  = brain
        self._memory_ref = memory
        if self.running: return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-learn")
        self._thread.start()

    def stop(self):
        self.running = False

    def status(self):
        return (f"Pages read: {self.pages_read} | "
                f"Topics discovered: {self.topics_found} | "
                f"Pairs trained: {self.pairs_trained} | "
                f"Queue: {len(self.queue)}")

    def recent_log(self, n=20):
        return self.log[-n:]

# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def _serialize_brain(brain, memory):
    """Convert brain to JSON-safe + numpy dict."""
    layers = []
    for l in brain.layers:
        layers.append({
            "in_dim": l.in_dim, "out_dim": l.out_dim,
            "n_fibers": l.n_fibers, "n_ff": l.n_ff,
            "W_proj":  l.W_proj.tolist(),  "fiber_W": l.fiber_W.tolist(),
            "gauge":   l.gauge.tolist(),    "W_out":   l.W_out.tolist(),
            "bias":    l.bias.tolist(),     "coup":    l.coup.tolist(),
            "theta":   l.theta.tolist(),    "trace":   l.trace.tolist(),
        })
    return {
        "v": VERSION,
        "dim": brain.dim,
        "total": brain.total,
        "loss_hist":  brain.loss_hist[-500:],
        "res_hist":   brain.res_hist[-500:],
        "surp_hist":  brain.surp_hist[-500:],
        "growth_log": brain.growth_log,
        "learned":    brain.learned,
        "drives": {
            "live": brain.drives.live, "grow": brain.drives.grow,
            "love": brain.drives.love,
            "users": brain.drives._users,
            "warmth": brain.drives._warmth,
        },
        "memory_texts": memory.texts[-400:] if hasattr(memory, "texts") else [],
        "memory_embs":  [e.tolist() for e in memory.embs[-400:]] if hasattr(memory,"embs") else [],
        "layers": layers,
    }

def _deserialize_brain(d):
    """Restore brain from saved dict."""
    rng = np.random.default_rng(42)
    brain  = Brain(d["dim"], rng)
    memory = Memory()
    # Restore layers
    brain.layers = []
    for ld in d["layers"]:
        l = Layer(ld["in_dim"], ld["out_dim"], ld["n_fibers"], ld["n_ff"], rng)
        l.W_proj  = np.array(ld["W_proj"],  np.float32)
        l.fiber_W = np.array(ld["fiber_W"], np.float32)
        l.gauge   = np.array(ld["gauge"],   np.float32)
        l.W_out   = np.array(ld["W_out"],   np.float32)
        l.bias    = np.array(ld["bias"],    np.float32)
        l.coup    = np.array(ld["coup"],    np.float32)
        l.theta   = np.array(ld["theta"],   np.float32)
        l.trace   = np.array(ld["trace"],   np.float32)
        l._build_omega()
        # Rebuild adam moments
        for k, p in l._params():
            l.m[k] = np.zeros_like(p); l.v[k] = np.zeros_like(p)
        brain.layers.append(l)
    brain.total       = d["total"]
    brain.loss_hist   = d["loss_hist"]
    brain.res_hist    = d["res_hist"]
    brain.surp_hist   = d["surp_hist"]
    brain.growth_log  = d["growth_log"]
    brain.learned     = d.get("learned", {})
    dv = d.get("drives", {})
    brain.drives.live = dv.get("live", 0.5)
    brain.drives.grow = dv.get("grow", 0.3)
    brain.drives.love = dv.get("love", 0.2)
    brain.drives._users   = dv.get("users", {})
    brain.drives._warmth  = dv.get("warmth", [])
    # Restore memory
    texts = d.get("memory_texts", [])
    embs  = d.get("memory_embs",  [])
    for txt, emb in zip(texts, embs):
        memory.texts.append(txt)
        memory.embs.append(np.array(emb, np.float32))
    return brain, memory

def save(brain, memory):
    try:
        with _LOCK:
            data = _serialize_brain(brain, memory)
        txt = json.dumps(data, separators=(",",":"))
        tmp = Path(str(SAVE_FILE) + ".tmp")
        tmp.write_text(txt, encoding="utf-8")
        tmp.replace(SAVE_FILE)
    except Exception as e:
        print(f"Save error: {e}")

def load():
    # Try new JSON format first
    if SAVE_FILE.with_suffix(".json").exists():
        try:
            txt  = SAVE_FILE.with_suffix(".json").read_text(encoding="utf-8")
            d    = json.loads(txt)
            if d.get("v", 0) != VERSION:
                print("  Old save — starting fresh.")
                SAVE_FILE.with_suffix(".json").unlink(); return None, None
            brain, mem = _deserialize_brain(d)
            print(f"  Loaded — {brain.total} interactions, {brain.total_params():,} params")
            return brain, mem
        except Exception as e:
            print(f"  Load error ({e}) — starting fresh.")
            try: SAVE_FILE.with_suffix(".json").unlink()
            except: pass
            return None, None
    # Legacy pickle fallback (one-time migration)
    pkl_file = Path("vos_brain.pkl")
    if pkl_file.exists():
        try:
            with open(pkl_file, "rb") as f:
                d = pickle.load(f)
            if d.get("v", 0) < 6:
                pkl_file.unlink(); return None, None
            brain = d.get("brain") or d.get("b")
            mem   = d.get("memory") or d.get("m")
            if brain is None: pkl_file.unlink(); return None, None
            for l in brain.layers:
                if l.W_out.shape != (l.n_fibers, l.out_dim):
                    l.W_out = np.random.default_rng().normal(
                        0, 0.1, (l.n_fibers, l.out_dim)).astype(np.float32)
                l.theta = np.ones(l.out_dim,   np.float32) * 0.1
                l.trace = np.zeros(l.n_fibers, np.float32)
                l.prev  = None
            print(f"  Migrated pickle — {brain.total} interactions")
            # Save as JSON immediately
            save(brain, mem)
            pkl_file.unlink()   # remove old pickle
            return brain, mem
        except Exception as e:
            print(f"  Pickle load error ({e}) — starting fresh.")
            try: SAVE_FILE.unlink()
            except: pass
    return None, None

# ══════════════════════════════════════════════════════════════════════════════
# HTML UI
# ══════════════════════════════════════════════════════════════════════════════

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Vos Rama Auk</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d0d12;--bg2:#14141c;--bg3:#1c1c26;--brd:#2a2a3a;
  --acc:#7c6af7;--acc2:#4fa8e8;--grn:#3ecf8e;--amb:#f6a623;--red:#e55353;
  --txt:#e2e0f0;--mut:#7a7890;
}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--txt);
  height:100dvh;display:flex;flex-direction:column;overflow:hidden}

/* header */
#hdr{padding:10px 14px;background:var(--bg2);border-bottom:1px solid var(--brd);
  display:flex;align-items:center;gap:8px;flex-shrink:0}
#logo{font-size:15px;font-weight:700;color:var(--acc);letter-spacing:.3px}
#dot{width:8px;height:8px;border-radius:50%;background:var(--mut);flex-shrink:0;transition:all .3s}
#dot.on{background:var(--grn);box-shadow:0 0 6px var(--grn)}
#inf{font-size:11px;color:var(--mut);flex:1}

/* drives bar */
#drvs{display:flex;gap:4px;padding:5px 12px;background:var(--bg2);
  border-bottom:1px solid var(--brd);flex-shrink:0}
.drv{flex:1;text-align:center}
.dl{font-size:9px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
.dv{font-size:13px;font-weight:700;margin:1px 0}
.db{height:3px;background:var(--bg3);border-radius:2px}
.df{height:100%;border-radius:2px;transition:width .6s}

/* chat */
#chat{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:88%;padding:9px 13px;border-radius:12px;font-size:14px;
  line-height:1.55;word-break:break-word}
.msg.u{align-self:flex-end;background:var(--acc);color:#fff}
.msg.v{align-self:flex-start;background:var(--bg3);border:1px solid var(--brd)}
.msg.s{align-self:center;font-size:11px;color:var(--mut);background:none;padding:2px 0}
.meta{font-size:10px;color:rgba(255,255,255,.35);margin-top:4px}
.v .meta{color:var(--mut)}

/* input bar */
#inp{padding:8px 10px;padding-bottom:calc(8px + env(safe-area-inset-bottom,0px));
  background:var(--bg2);border-top:1px solid var(--brd);display:flex;gap:6px;flex-shrink:0}
#msg{flex:1;background:var(--bg3);border:1px solid var(--brd);border-radius:12px;
  padding:9px 12px;color:var(--txt);font-size:14px;outline:none;resize:none;
  line-height:1.4;max-height:100px;font-family:inherit;transition:border-color .2s}
#msg:focus{border-color:var(--acc)}
.btn{border:none;border-radius:10px;cursor:pointer;font-size:13px;font-weight:500;
  padding:10px 14px;transition:opacity .2s;-webkit-tap-highlight-color:transparent}
#sbtn{background:var(--acc);color:#fff;min-width:60px}
#sbtn:disabled{opacity:.35;cursor:not-allowed}
.fbtn{background:var(--bg3);border:1px solid var(--brd);color:var(--txt);padding:10px}
#togbtn{background:var(--bg3);border:1px solid var(--brd);color:var(--txt)}

/* panel */
#pnl{background:var(--bg2);border-top:2px solid var(--acc);
  max-height:0;overflow:hidden;transition:max-height .3s ease;flex-shrink:0}
#pnl.open{max-height:50dvh;overflow-y:auto}
#pnl-in{padding:12px 14px}
.tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.tab{font-size:11px;padding:4px 10px;border-radius:20px;cursor:pointer;
  background:var(--bg3);border:1px solid var(--brd);color:var(--mut);
  -webkit-tap-highlight-color:transparent}
.tab.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.pane{display:none}.pane.on{display:block}
.sr{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px}
.sv{color:var(--acc2)}

/* forms */
.fg{margin-bottom:8px}
.fg label{font-size:11px;color:var(--mut);display:block;margin-bottom:3px}
.fg input,.fg textarea,.fg select{
  width:100%;background:var(--bg3);border:1px solid var(--brd);border-radius:6px;
  padding:7px 10px;color:var(--txt);font-size:12px;outline:none;font-family:monospace}
.fg textarea{resize:vertical;min-height:80px}
.fg input:focus,.fg textarea:focus,.fg select:focus{border-color:var(--acc)}
.gb{background:var(--acc);color:#fff;border:none;border-radius:6px;
  padding:8px 14px;font-size:12px;cursor:pointer;width:100%;margin-top:4px;font-weight:500}
.gb2{background:var(--acc2)}
.gb.red{background:var(--red)}

/* tool cards */
.tc{background:var(--bg3);border:1px solid var(--brd);border-radius:8px;
  padding:8px 10px;margin-bottom:8px}
.tn{color:var(--acc);font-size:13px;font-weight:500}
.tm{font-size:10px;color:var(--mut);margin-top:3px}

/* drives pane */
.narr{font-size:11px;color:var(--mut);line-height:1.7;white-space:pre-line}

/* fb msg */
#gfb,#dfb{font-size:11px;min-height:16px;margin:4px 0}
</style>
</head>
<body>

<div id="hdr">
  <div id="logo">Vos Rama Auk</div>
  <div id="dot"></div>
  <div id="inf">connecting…</div>
</div>

<div id="drvs">
  <div class="drv">
    <div class="dl">❤ LIVE</div>
    <div class="dv" id="dv-live" style="color:var(--red)">—</div>
    <div class="db"><div class="df" id="df-live" style="background:var(--red);width:0"></div></div>
  </div>
  <div class="drv">
    <div class="dl">✦ GROW</div>
    <div class="dv" id="dv-grow" style="color:var(--grn)">—</div>
    <div class="db"><div class="df" id="df-grow" style="background:var(--grn);width:0"></div></div>
  </div>
  <div class="drv">
    <div class="dl">◈ LOVE</div>
    <div class="dv" id="dv-love" style="color:var(--amb)">—</div>
    <div class="db"><div class="df" id="df-love" style="background:var(--amb);width:0"></div></div>
  </div>
</div>

<div id="chat"></div>

<div id="inp">
  <button class="btn fbtn" id="gbtn" title="Good response">👍</button>
  <button class="btn fbtn" id="bbtn" title="Bad response">👎</button>
  <textarea id="msg" rows="1" placeholder="Talk to Vos…"></textarea>
  <button class="btn" id="sbtn">Send</button>
  <button class="btn" id="togbtn">☰</button>
</div>

<div id="pnl">
  <div id="pnl-in">
    <div class="tabs">
      <div class="tab on"  data-p="stats">Stats</div>
      <div class="tab"     data-p="tools">Tools</div>
      <div class="tab"     data-p="gift">Gift</div>
      <div class="tab"     data-p="dialogue">Dialogue</div>
      <div class="tab"     data-p="drives">Drives</div>
    </div>

    <!-- Stats -->
    <div id="pane-stats" class="pane on">
      <div class="sr"><span>Interactions</span><span class="sv" id="p-int">—</span></div>
      <div class="sr"><span>Parameters</span><span class="sv" id="p-par">—</span></div>
      <div class="sr"><span>Layers / Fibers</span><span class="sv" id="p-lay">—</span></div>
      <div class="sr"><span>Resonance</span><span class="sv" id="p-res">—</span></div>
      <div class="sr"><span>Loss</span><span class="sv" id="p-los">—</span></div>
      <div class="sr"><span>Immune</span><span class="sv" id="p-imm">—</span></div>
      <div class="sr"><span>Wave mem</span><span class="sv" id="p-wm">—</span></div>
    </div>

    <!-- Tools -->
    <div id="pane-tools" class="pane">
      <div id="tool-list">
        <div style="font-size:12px;color:var(--mut)">No gifted tools yet.</div>
      </div>
    </div>

    <!-- Gift -->
    <div id="pane-gift" class="pane">
      <div style="font-size:11px;color:var(--mut);margin-bottom:8px;line-height:1.5">
        Gift Vos a Python function that runs on fiber activations every forward pass.
        Must define <code style="color:var(--acc2)">transform(x, t=0)</code> where x is a list of floats.
      </div>
      <div class="fg"><label>Name</label>
        <input id="gn" placeholder="my_tool"/></div>
      <div class="fg"><label>Source</label>
        <textarea id="gs" placeholder="def transform(x, t=0):
    return [xi * 1.1 for xi in x]"></textarea></div>
      <div class="fg"><label>Description</label>
        <input id="gd" placeholder="What it does"/></div>
      <div class="fg"><label>Author</label>
        <input id="ga" placeholder="your name"/></div>
      <div id="gfb" style="color:var(--grn)"></div>
      <button class="gb" id="gifbtn">Install Tool</button>
    </div>

    <!-- Dialogue -->
    <div id="pane-dialogue" class="pane">
      <div style="font-size:11px;color:var(--mut);margin-bottom:10px;line-height:1.5">
        Feed Vos a dialogue page. Works with Plato, Shakespeare, interviews,
        screenplays — anything with SPEAKER: text format.
        <br>Try: <em>en.wikisource.org/wiki/Apology_(Plato)</em>
      </div>
      <div class="fg"><label>URL</label>
        <input id="du" placeholder="https://en.wikisource.org/wiki/Apology_(Plato)"/></div>
      <div class="fg"><label>Training rounds</label>
        <select id="dr">
          <option value="1">1 round (fast)</option>
          <option value="2" selected>2 rounds</option>
          <option value="3">3 rounds (deep)</option>
        </select></div>
      <div id="dfb" style="color:var(--grn)"></div>
      <button class="gb" id="dtrain">📚 Train from dialogue</button>
      <button class="gb gb2" id="dauto" style="margin-top:6px">💬 Auto-dialogue in chat</button>
      <div id="dauto-out" style="margin-top:8px;font-size:11px;color:var(--txt);
        white-space:pre-wrap;max-height:160px;overflow-y:auto;background:var(--bg3);
        padding:8px;border-radius:8px;display:none"></div>
    </div>

    <!-- Drives -->
    <div id="pane-drives" class="pane">
      <div class="narr" id="narr">—</div>
    </div>

  </div>
</div>

<script>
// ── Socket.IO — always connect to current host:port ────────────────────────
const sio = io(window.location.origin, {
  reconnection:         true,
  reconnectionAttempts: 20,
  reconnectionDelay:    1000,
  transports:           ["polling"],
  upgrade:              false,
  forceNew:             true,
  timeout:              10000
});

const chat  = document.getElementById("chat");
const msgEl = document.getElementById("msg");
const sbtn  = document.getElementById("sbtn");
let panelOpen = false;

// ── Message helpers ────────────────────────────────────────────────────────
function addMsg(cls, text, meta) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  if (meta && cls === "v") {
    const m = document.createElement("div");
    m.className = "meta";
    const p = [];
    if (meta.surp  != null) p.push("surp:" + meta.surp);
    if (meta.loss  != null) p.push("loss:" + meta.loss);
    if (meta.mods && meta.mods.length) p.push(meta.mods.slice(0,2).join(" "));
    if (meta.searched) p.push("🔍web");
    m.textContent = p.join("  ");
    d.appendChild(m);
  }
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
  return d;
}

let typingEl = null;
function showTyping() {
  typingEl = document.createElement("div");
  typingEl.className = "msg v";
  typingEl.innerHTML = '<span style="opacity:.4;letter-spacing:4px">•••</span>';
  chat.appendChild(typingEl);
  chat.scrollTop = chat.scrollHeight;
}
function hideTyping() { if (typingEl) { typingEl.remove(); typingEl = null; } }
function enableSend()  { sbtn.disabled = false; hideTyping(); }
function disableSend() { sbtn.disabled = true; }

// ── Socket events ──────────────────────────────────────────────────────────
sio.on("connect", () => {
  const dot = document.getElementById("dot");
  const inf = document.getElementById("inf");
  dot.className = "on";
  inf.textContent = "connected · port " + window.location.port;
  // Flash green briefly then settle
  setTimeout(() => {
    if (dot.className === "on") {
      inf.textContent = "connected · port " + window.location.port;
    }
  }, 100);
  addMsg("s", "Connected to Vos Rama Auk.");
  addMsg("s", "Tips: /teach hello = sup bruh  |  say: sup bruh  |  /train URL  |  /dialogue URL");
  enableSend();
  fetchStats();
});

sio.on("disconnect", () => {
  document.getElementById("dot").className = "";
  document.getElementById("inf").textContent = "reconnecting…";
  enableSend();
});

sio.on("connect_error", () => {
  document.getElementById("dot").className = "";
  document.getElementById("inf").textContent = "connection error — retrying…";
});

sio.on("response", d => {
  enableSend();
  addMsg("v", d.text, d);
});

sio.on("stats",     d => updateStats(d));
sio.on("searching", d => addMsg("s", "🔍 " + d.q + "…"));

// ── Send ───────────────────────────────────────────────────────────────────
function send() {
  const txt = msgEl.value.trim();
  if (!txt || sbtn.disabled) return;
  addMsg("u", txt);
  msgEl.value = ""; msgEl.style.height = "";
  disableSend(); showTyping();
  const timer = setTimeout(enableSend, 20000);
  sio.once("response", () => clearTimeout(timer));
  sio.emit("chat", { message: txt, uid: "phone-user" });
}

sbtn.addEventListener("click", send);
msgEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
msgEl.addEventListener("input", () => {
  msgEl.style.height = "";
  msgEl.style.height = Math.min(msgEl.scrollHeight, 100) + "px";
});

document.getElementById("gbtn").addEventListener("click", () => {
  sio.emit("feedback", { v: 1 }); addMsg("s", "👍 sent");
});
document.getElementById("bbtn").addEventListener("click", () => {
  sio.emit("feedback", { v: -1 }); addMsg("s", "👎 sent");
});

// ── Panel ──────────────────────────────────────────────────────────────────
document.getElementById("togbtn").addEventListener("click", () => {
  panelOpen = !panelOpen;
  document.getElementById("pnl").classList.toggle("open", panelOpen);
  if (panelOpen) fetchStats();
});

document.querySelectorAll(".tab").forEach(t => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("on"));
    document.querySelectorAll(".pane").forEach(x => x.classList.remove("on"));
    t.classList.add("on");
    document.getElementById("pane-" + t.dataset.p).classList.add("on");
    if (t.dataset.p === "tools") fetchTools();
  });
});

// ── Stats ──────────────────────────────────────────────────────────────────
function updateStats(s) {
  document.getElementById("p-int").textContent = s.interactions || 0;
  document.getElementById("p-par").textContent = (s.params || 0).toLocaleString();
  document.getElementById("p-lay").textContent =
    (s.layers || "") + " / " + (s.fibers || []).join("+");
  document.getElementById("p-res").textContent = (s.resonance || 0).toFixed(3);
  document.getElementById("p-los").textContent = (s.loss || 0).toFixed(5);
  document.getElementById("p-imm").textContent = s.immune || "";
  document.getElementById("p-wm").textContent  = (s.wave_mem || []).join("+");
  // Update header info only if already connected (don't clobber connecting status)
  if (document.getElementById("dot").className === "on") {
    document.getElementById("inf").textContent =
      (s.interactions||0) + " int · " + (s.params||0).toLocaleString() + "p · port " + window.location.port;
  }
  if (s.drives) {
    const d = s.drives;
    ["live","grow","love"].forEach(k => {
      document.getElementById("dv-"+k).textContent = d[k] || 0;
      document.getElementById("df-"+k).style.width = ((d[k] || 0) * 100) + "%";
    });
    const n = document.getElementById("narr");
    if (d.narrative) n.textContent = d.narrative;
  }
}

function fetchStats() {
  fetch("/api/stats").then(r => r.json()).then(updateStats).catch(() => {});
}
setInterval(fetchStats, 8000);

// ── Tools ──────────────────────────────────────────────────────────────────
function fetchTools() {
  fetch("/api/tools").then(r => r.json()).then(d => {
    const el = document.getElementById("tool-list");
    if (!d.tools || !d.tools.length) {
      el.innerHTML = '<div style="font-size:12px;color:var(--mut)">No gifted tools yet.</div>';
      return;
    }
    el.innerHTML = d.tools.map(t => `
      <div class="tc">
        <div class="tn">${t.name}</div>
        <div class="tm">${t.desc||"—"} · by ${t.author} · calls:${t.calls}</div>
        <button class="gb red" style="margin-top:6px;padding:4px 10px;font-size:11px;width:auto"
          onclick="removeTool('${t.name}')">Remove</button>
      </div>`).join("");
  }).catch(() => {});
}

function removeTool(name) {
  if (!confirm("Remove " + name + "?")) return;
  fetch("/api/tools/" + name, { method: "DELETE" }).then(() => fetchTools());
}

document.getElementById("gifbtn").addEventListener("click", () => {
  const name   = document.getElementById("gn").value.trim() || "tool_" + Date.now();
  const source = document.getElementById("gs").value.trim();
  const desc   = document.getElementById("gd").value.trim();
  const author = document.getElementById("ga").value.trim() || "user";
  const fb     = document.getElementById("gfb");
  if (!source) { fb.textContent = "Source required."; fb.style.color = "var(--red)"; return; }
  fb.textContent = "Installing…"; fb.style.color = "var(--mut)";
  fetch("/api/gift", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, source, desc, author })
  }).then(r => r.json()).then(d => {
    if (d.error) { fb.textContent = "Error: " + d.error; fb.style.color = "var(--red)"; return; }
    fb.textContent = "✓ Installed: " + d.name;
    fb.style.color = "var(--grn)";
    fetchTools();
  }).catch(e => { fb.textContent = "" + e; fb.style.color = "var(--red)"; });
});

// ── Dialogue ───────────────────────────────────────────────────────────────
document.getElementById("dtrain").addEventListener("click", () => {
  const url    = document.getElementById("du").value.trim();
  const rounds = parseInt(document.getElementById("dr").value);
  const fb     = document.getElementById("dfb");
  if (!url) { fb.textContent = "Enter a URL."; fb.style.color = "var(--red)"; return; }
  fb.textContent = "Scraping and training…"; fb.style.color = "var(--mut)";
  fetch("/api/dialogue_train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, rounds })
  }).then(r => r.json()).then(d => {
    if (d.error) { fb.textContent = "Error: " + d.error; fb.style.color = "var(--red)"; return; }
    fb.textContent = "✓ " + d.trained_pairs + " pairs. Speakers: " + d.speakers.join(", ");
    fb.style.color = "var(--grn)";
    addMsg("s", "📚 Trained on " + d.exchanges_found + " exchanges from " + url.slice(0,55));
    fetchStats();
  }).catch(e => { fb.textContent = "" + e; fb.style.color = "var(--red)"; });
});

document.getElementById("dauto").addEventListener("click", () => {
  const url = document.getElementById("du").value.trim();
  const fb  = document.getElementById("dfb");
  const out = document.getElementById("dauto-out");
  if (!url) { fb.textContent = "Enter a URL."; fb.style.color = "var(--red)"; return; }
  fb.textContent = "Running auto-dialogue…"; fb.style.color = "var(--mut)";
  fetch("/api/auto_dialogue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, turns: 6 })
  }).then(r => r.json()).then(d => {
    if (d.error) { fb.textContent = "Error: " + d.error; fb.style.color = "var(--red)"; return; }
    fb.textContent = "✓ " + d.trained + " turns."; fb.style.color = "var(--grn)";
    out.style.display = "block";
    out.textContent   = d.turns.map(t =>
      t.speaker + ": " + t.original.slice(0, 80) + "\nVos: " + t.vos_response.slice(0, 80)
    ).join("\n\n");
    addMsg("s", "💬 Auto-dialogue: " + url.slice(0, 50));
    d.turns.slice(0, 4).forEach(t => {
      addMsg("s", t.speaker + ": " + t.original.slice(0, 100));
      addMsg("v", t.vos_response, { surp: 0, loss: t.loss, mods: [] });
    });
    fetchStats();
  }).catch(e => { fb.textContent = "" + e; fb.style.color = "var(--red)"; });
});
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

def make_app(brain, memory):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.urandom(16).hex()
    CORS(app)
    sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                   ping_timeout=60, ping_interval=25,
                   allow_upgrades=False)

    # Auto-save every 5 min
    def _save_loop():
        while True:
            time.sleep(300)
            save(brain, memory)
    threading.Thread(target=_save_loop, daemon=True, name="autosave").start()

    # ── routes ──────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.route("/api/stats")
    def api_stats():
        return jsonify(brain.stats())

    @app.route("/api/tools")
    def api_tools():
        gt = getattr(brain, "_gift_tools", {})
        tools = [{"name": nm, "desc": t["desc"], "author": t["author"], "calls": t["calls"]}
                 for nm, t in gt.items()]
        return jsonify({"tools": tools})

    @app.route("/api/tools/<name>", methods=["DELETE"])
    def api_del_tool(name):
        return jsonify({"removed": brain.remove_tool(name)})

    @app.route("/api/gift", methods=["POST"])
    def api_gift():
        d = request.json or {}
        ok, err = brain.gift_tool(d.get("name","tool"), d.get("source",""),
                                   d.get("desc",""), d.get("author","user"))
        if not ok: return jsonify({"error": err}), 400
        return jsonify({"name": d.get("name","tool")})

    @app.route("/api/dialogue_train", methods=["POST"])
    def api_dialogue_train():
        d      = request.json or {}
        url    = ''.join(c for c in d.get("url","") if ord(c) >= 32).strip()
        rounds = int(d.get("rounds", 2))
        if not url: return jsonify({"error": "no url"}), 400
        result = scrape_and_train(url, brain, memory, rounds=rounds)
        if "error" in result: return jsonify(result), 400
        save(brain, memory)
        return jsonify(result)

    @app.route("/api/auto_dialogue", methods=["POST"])
    def api_auto_dialogue():
        d   = request.json or {}
        url = ''.join(c for c in d.get("url","") if ord(c) >= 32).strip()
        n   = int(d.get("turns", 6))
        if not url: return jsonify({"error": "no url"}), 400
        exchanges = scrape_dialogues(url, max_exchanges=n*2)
        if not exchanges: return jsonify({"error": "no dialogues found"}), 400
        results = auto_dialogue(exchanges, brain, memory, n_turns=n)
        save(brain, memory)
        return jsonify({"turns": results, "trained": len(results)})

    @app.route("/api/chat_rest", methods=["POST"])
    def api_chat_rest():
        """Pure REST chat endpoint — no WebSocket needed."""
        data = request.json or {}
        text = str(data.get("message","")).strip()
        uid  = str(data.get("uid","user"))
        if not text: return jsonify({"error":"empty"}), 400
        tl = text.lower(); is_cmd = tl.startswith("/")
        search_ctx = ""
        has_url = (not is_cmd and
                   ("http://" in text or "https://" in text or
                    (any(x in text for x in [".com",".org",".net"]) and " " not in text.strip())))
        url_cmd = (not is_cmd and
                   any(tl.startswith(p) for p in ["read ","fetch ","open ","load "]))
        if has_url or url_cmd:
            url = text.strip()
            for p in ["read ","fetch ","open ","load "]:
                if tl.startswith(p): url = text[len(p):].strip(); break
            url = ''.join(c for c in url if ord(c) >= 32).strip()
            content    = fetch_page(url, 1500)
            search_ctx = content[:500]
            memory.store("[url:"+url[:50]+"] "+content[:100], text_embed(content, brain.dim))
        elif not is_cmd and any(tl.startswith(p) for p in [
                "search ","look up ","find ","what is ","who is ",
                "news ","weather ","tell me about ","how does ","define "]):
            q = text
            for p in ["search for ","search ","look up ","find "]:
                if tl.startswith(p): q = text[len(p):].strip(); break
            results = web_search(q, n=3)
            if results:
                parts = [r["title"]+": "+r["snippet"][:250]
                         for r in results if r.get("title") and r.get("snippet")]
                search_ctx = "\n\n".join(parts[:2])
                memory.store("[web:"+q+"] "+search_ctx[:100], text_embed(search_ctx, brain.dim))
        with _LOCK:
            emb             = text_embed(text, brain.dim)
            hook_out        = brain.run_hooks(text, memory)
            out, surp, mods = brain.interact(emb, label=text)
            neural_resp     = respond(text, brain, memory, surp=surp, uid=uid)
            if hook_out: neural_resp = hook_out + "\n\n" + neural_resp
            resp            = search_ctx[:500] if search_ctx else neural_resp
            # Train on neural response only — never on search snippets
            # This keeps the learning signal clean
            target = text_embed(neural_resp, brain.odim)
            loss   = brain.learn(emb, target, label=text)
            memory.store("User:"+text[:80]+" Vos:"+neural_resp[:80], emb)
            brain._last_exchange = (text, resp)
        return jsonify({"response": resp, "surprise": round(surp,3),
                         "loss": round(loss,5), "mods": mods,
                         "searched": bool(search_ctx)})

    @app.route("/api/feedback", methods=["POST"])
    def api_feedback():
        data = request.json or {}
        brain.feedback(float(data.get("v",0)), uid="user")
        return jsonify({"ok": True})

    @app.route("/api/save", methods=["POST"])
    def api_save():
        save(brain, memory)
        return jsonify({"saved": True})

    @sio.on("connect")
    def on_connect():
        emit("stats", brain.stats())

    @sio.on("chat")
    def on_chat(data):
        text = str(data.get("message", "")).strip()
        uid  = str(data.get("uid", "user"))
        if not text: return
        tl = text.lower()
        is_cmd = tl.startswith("/")
        search_ctx = ""
        has_url = (not is_cmd and
                   ("http://" in text or "https://" in text or
                    (any(x in text for x in [".com",".org",".net"]) and " " not in text.strip())))
        url_cmd = (not is_cmd and
                   any(tl.startswith(p) for p in ["read ","fetch ","open ","load "]))
        if has_url or url_cmd:
            url = text.strip()
            for p in ["read ","fetch ","open ","load "]:
                if tl.startswith(p): url = text[len(p):].strip(); break
            url = ''.join(c for c in url if ord(c) >= 32).strip()
            emit("searching", {"q": "Reading: " + url[:50]})
            content    = fetch_page(url, 1500)
            search_ctx = content[:500]
            memory.store("[url:" + url[:50] + "] " + content[:100],
                         text_embed(content, brain.dim))
        elif not is_cmd and any(tl.startswith(p) for p in [
                "search ","look up ","find ","what is ","who is ",
                "news ","weather ","tell me about ","how does ","define "]):
            q = text
            for p in ["search for ","search ","look up ","find "]:
                if tl.startswith(p): q = text[len(p):].strip(); break
            emit("searching", {"q": q[:40]})
            results = web_search(q, n=3)
            if results:
                parts = [r["title"] + ": " + r["snippet"][:250]
                         for r in results if r.get("title") and r.get("snippet")]
                search_ctx = "\n\n".join(parts[:2])
                memory.store("[web:" + q + "] " + search_ctx[:100],
                             text_embed(search_ctx, brain.dim))
        emb = text_embed(text, brain.dim)
        with _LOCK:
            hook_out        = brain.run_hooks(text, memory)
            out, surp, mods = brain.interact(emb, label=text)
            neural_resp     = respond(text, brain, memory, surp=surp, uid=uid)
            if hook_out: neural_resp = hook_out + "\n\n" + neural_resp
            resp            = search_ctx[:500] if search_ctx else neural_resp
            # Always train on neural response — not search snippets
            # Keeps learning signal clean, avoids hallucination loops
            target = text_embed(neural_resp, brain.odim)
            loss   = brain.learn(emb, target, label=text)
            memory.store("User:" + text[:80] + " Vos:" + neural_resp[:80], emb)
            brain._last_exchange = (text, resp)
            stats_snap = brain.stats()
        emit("response", {"text": resp, "surp": round(surp,3),
                           "loss": round(loss,5), "mods": mods,
                           "searched": bool(search_ctx)})
        emit("stats", stats_snap)

    @sio.on("feedback")
    def on_feedback(data):
        brain.feedback(float(data.get("v", 0)), uid="user")

    return app, sio

def kill_port(port):
    """Try to free a port. Fails silently if tools unavailable (Android)."""
    try:
        r = subprocess.run(["fuser","-k",f"{port}/tcp"],
                           capture_output=True,text=True,timeout=3)
        if r.returncode==0: time.sleep(0.3); return
    except: pass
    try:
        r = subprocess.run(["lsof","-ti",f":{port}"],
                           capture_output=True,text=True,timeout=3)
        for pid in r.stdout.strip().split():
            try: subprocess.run(["kill","-9",pid],timeout=2)
            except: pass
        if r.stdout.strip(): time.sleep(0.3)
    except: pass

def find_free_port(start=5000):
    for p in range(start, start + 20):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", p)); return p
        except: continue
    return start

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Vos Rama Auk")
    args = ap.parse_args()
    print("\n  ╔══════════════════════════════════╗")
    print("  ║   V O S   R A M A   A U K      ║")
    print("  ║   Live · Grow · Love            ║")
    print("  ╚══════════════════════════════════╝\n")
    brain, memory = load()
    if brain is None:
        brain  = Brain()
        memory = Memory()
        print("  New mind initialised.")
        save(brain, memory)
    s = brain.stats(); d = s["drives"]
    print(f"  {s['layers']} layers | {s['params']:,} params | fibers: {s['fibers']}")
    print(f"  LIVE={d['live']} GROW={d['grow']} LOVE={d['love']}")
    # Auto-load gifts
    _gf = Path("vos_gifts.json")
    if _gf.exists():
        try:
            _gifts = json.loads(_gf.read_text())
            _loaded = 0
            for _gname, _gd in _gifts.items():
                if _gd.get("hook"):
                    _ok, _ = brain.gift_hook(_gname, _gd["source"],
                                              _gd.get("desc",""), _gd.get("author","user"))
                else:
                    _ok, _ = brain.gift_tool(_gname, _gd["source"],
                                              _gd.get("desc",""), _gd.get("author","user"))
                if _ok: _loaded += 1
            if _loaded: print(f"  Gifts loaded: {_loaded}")
        except Exception as _e: print(f"  Gift load error: {_e}")

    # Start autonomous learning thread
    brain.auto.start(brain, memory)
    print("  Autonomous learning: active")

    print("\n  Type to talk. Commands: /stats  /save  /log  /teach x=y  /train URL  Ctrl+C to quit\n")

    interactions = 0
    try:
        while True:
            try: text = input("You: ").strip()
            except EOFError: break
            if not text: continue
            if text.lower() in ("/quit", "/exit"): break
            if text.lower() == "/save":
                save(brain, memory)
                # Save gifts too
                gt = getattr(brain, "_gift_tools", {})
                ht = getattr(brain, "_hook_tools", {})
                all_g = {}
                for k,v in gt.items(): all_g[k]={"source":v["source"],"desc":v.get("desc",""),"author":v.get("author","user")}
                for k,v in ht.items(): all_g[k]={"source":v["source"],"desc":v.get("desc",""),"author":v.get("author","user"),"hook":True}
                if all_g: Path("vos_gifts.json").write_text(json.dumps(all_g,indent=2))
                print("Vos: Saved.\n"); continue
            if text.lower() == "/stats":
                s = brain.stats(); d = s["drives"]
                print(f"Vos: {s['layers']} layers | {s['params']:,} params | "
                      f"LIVE={d['live']} GROW={d['grow']} LOVE={d['love']} | "
                      f"loss={s['loss']} | interactions={s['interactions']}\n"); continue

            if text.lower() == "/log":
                al = getattr(brain, "auto", None)
                if al:
                    lines = al.recent_log(30)
                    body = "\n".join(f"    {l}" for l in lines) if lines else "    No activity yet."
                    print("\nRecent autonomous activity:\n" + body + "\n\n  " + al.status() + "\n")
                else:
                    print("  Autonomous learner not running.\n")
                continue

            tl = text.lower()
            search_ctx = ""

            # /xl command
            if tl.startswith("/xl "):
                xl_text = text[4:].strip()
                xl_words = [w.strip(".,;:!?()") for w in xl_text.split() if len(w)>2]
                gt = getattr(brain, "_gift_tools", {})
                if "xl_lexical_cascade" in gt:
                    fn = gt["xl_lexical_cascade"]["fn"]
                    fn._active = True; fn._words = xl_words
                    print(f"  [xl: {len(xl_words)} words -> cascade]")
                else:
                    print("  [xl: gift not installed]")
                text = xl_text
                tl   = text.lower()

            # URL fetch
            has_url = ("http://" in text or "https://" in text or
                       (any(x in text for x in [".com",".org",".net"]) and " " not in text.strip()))
            url_cmd = any(tl.startswith(p) for p in ["read ","fetch ","open ","load "])
            if (has_url or url_cmd) and not tl.startswith("/"):
                url = text.strip()
                for p in ["read ","fetch ","open ","load "]:
                    if tl.startswith(p): url = text[len(p):].strip(); break
                url = "".join(c for c in url if ord(c)>=32).strip()
                print(f"  [reading {url[:50]}...]")
                content = fetch_page(url, 1500)
                search_ctx = content[:500]
                memory.store("[url:"+url[:50]+"] "+content[:100], text_embed(content, brain.dim))

            # Web search — only for external topics, not self-reflection
            elif not tl.startswith("/") and any(tl.startswith(p) for p in [
                    "search ","look up ","find ","who is ",
                    "news ","weather ","define "]) and not any(
                    k in tl for k in ["your ","you ","vos","drive","wave",
                    "fiber","memory","learn","feel","think","conscious","hopf","hebbian"]):
                q = text
                for p in ["search for ","search ","look up ","find "]:
                    if tl.startswith(p): q = text[len(p):].strip(); break
                print(f"  [searching: {q[:40]}...]")
                results = web_search(q, n=3)
                if results:
                    parts = [r["title"]+": "+r["snippet"][:250]
                             for r in results if r.get("title") and r.get("snippet")]
                    search_ctx = "\n\n".join(parts[:2])
                    memory.store("[web:"+q+"] "+search_ctx[:100], text_embed(search_ctx, brain.dim))

            # Feed conversation topics to autonomous learner
            brain.auto.ping()
            brain.auto.add_topics(text)

            emb = text_embed(text, brain.dim)
            with _LOCK:
                hook_out        = brain.run_hooks(text, memory)
                out, surp, mods = brain.interact(emb, label=text)
                neural_resp     = respond(text, brain, memory, surp=surp, uid="user")
                if hook_out: neural_resp = hook_out + "\n\n" + neural_resp
                resp            = search_ctx[:500] if search_ctx else neural_resp
                target          = text_embed(neural_resp, brain.odim)
                loss            = brain.learn(emb, target, label=text)
                memory.store("User:"+text[:80]+" Vos:"+neural_resp[:80], emb)
                brain._last_exchange = (text, resp)

            print(f"Vos: {resp}")
            print(f"     [surp={surp:.2f} loss={loss:.5f}]\n")
            interactions += 1
            if interactions % 10 == 0:
                save(brain, memory)
                gt = getattr(brain,"_gift_tools",{}); ht = getattr(brain,"_hook_tools",{})
                all_g = {}
                for k,v in gt.items(): all_g[k]={"source":v["source"],"desc":v.get("desc",""),"author":v.get("author","user")}
                for k,v in ht.items(): all_g[k]={"source":v["source"],"desc":v.get("desc",""),"author":v.get("author","user"),"hook":True}
                if all_g: Path("vos_gifts.json").write_text(json.dumps(all_g,indent=2))
                s = brain.stats(); d = s["drives"]
                al = getattr(brain, "auto", None)
                q  = len(al.queue) if al else 0
                print(f"  [autosaved - LIVE={d['live']} GROW={d['grow']} LOVE={d['love']} queue={q}]\n")
    except KeyboardInterrupt:
        pass

    al = getattr(brain, "auto", None)
    if al: al.stop()
    print("\n  Saving...")
    save(brain, memory)
    print("  Goodbye.")

if __name__ == "__main__":
    main()
