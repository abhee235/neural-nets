"""
Chapter 17 — BPE Tokenizer: all three figures.

Teaches:
  merge-rounds.svg   BPE training one merge at a time; the corpus shrinks each
                     round, and each round counts a corpus the last one rewrote.
  merge-tree.svg     merges compose — a new token is built from tokens that
                     earlier merges produced.
  order-matters.svg  the same rules in the wrong order give a worse tokenization,
                     because a rule consumes what an earlier rule produced.

Output:
  docs/assets/ch-17/merge-rounds.svg
  docs/assets/ch-17/merge-tree.svg
  docs/assets/ch-17/order-matters.svg

Every frame, count and token sequence below is COMPUTED by the BPE
implementation in this file, never typed. Run from the repo root:

    python scripts/media/ch-17-bpe-figures.py
"""
import io
import os
import sys
import xml.dom.minidom as minidom

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
NL = chr(10)
OUT = "docs/assets/ch-17/"
os.makedirs(OUT, exist_ok=True)

BG, INK, MUTE = "#faf7f0", "#2c2416", "#8b7355"
BLUE_F, BLUE_S = "#dde8f0", "#1e6091"
RED_F, RED_S = "#f3e6e4", "#a94442"
GRN = "#2e7d32"

# ─────────────────────────── the algorithm ───────────────────────────
# Mirrors src/tokenizer/bpe.ts. Kept here so the figures derive themselves.
CORPUS_TEXT = "low low low low low lower lower newest newest newest widest widest"


def count_pairs(corpus):
    """Adjacent pairs within each word. Never across the boundary between two."""
    counts = {}
    for seq in corpus:
        for i in range(len(seq) - 1):
            key = (seq[i], seq[i + 1])
            counts[key] = counts.get(key, 0) + 1
    return counts


def merge_pair(corpus, pair):
    """Replace the pair everywhere, advancing past BOTH halves on a match."""
    a, b = pair
    out = []
    for seq in corpus:
        new, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(seq[i])
                i += 1
        out.append(new)
    return out


def train(text, rounds):
    """Return the ordered merge rules and the corpus state after each one."""
    corpus = [list(w) for w in text.split(" ")]
    merges, states = [], [(None, corpus)]
    for _ in range(rounds):
        counts = count_pairs(corpus)
        # ties broken by first-seen; dict preserves insertion order, so this is
        # deterministic — the same hazard the chapter's section on ties covers
        best, best_n = None, 1
        for key, n in counts.items():
            if n > best_n:
                best_n, best = n, key
        if best is None:
            break
        corpus = merge_pair(corpus, best)
        merges.append(best)
        states.append((best, corpus))
    return merges, states


MERGES, STATES = train(CORPUS_TEXT, 8)
MERGE_NUMBER = {a + b: i + 1 for i, (a, b) in enumerate(MERGES)}

# ─────────────────────────── drawing helpers ───────────────────────────
WORDS = ["low", "lower", "newest", "widest"]
# index of the first occurrence of each distinct word, so a frame can show it
ALL_WORDS = CORPUS_TEXT.split(" ")
ROW_OF = [ALL_WORDS.index(w) for w in WORDS]


def esc(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def mono(x, y, t, size=10, fill=INK, anchor="start", weight="400"):
    return ('  <text x="%s" y="%s" text-anchor="%s" font-family="monospace" '
            'font-size="%s" font-weight="%s" fill="%s">%s</text>'
            % (x, y, anchor, size, weight, fill, t))


def tokbox(x, y, tok, w=None, h=20, hot=False, fs=11):
    w = w or max(22, int(fs * 0.68 * len(tok)) + 14)
    f, s = (RED_F, RED_S) if hot else (BLUE_F, BLUE_S)
    return ([
        '  <rect x="%s" y="%s" width="%s" height="%s" rx="4" fill="%s" '
        'stroke="%s" stroke-width="%s"/>' % (x, y, w, h, f, s, "2" if hot else "1.3"),
        mono(x + w / 2.0, y + h - 6, esc(tok), fs, INK, "middle", "700" if hot else "400"),
    ], w)


def write(name, body, note):
    open(OUT + name, "w", encoding="utf-8").write(body)
    minidom.parse(OUT + name)          # fails loudly if the XML is malformed
    print("%-20s %s" % (name, note))


# ════════════════ FIGURE 1 — merge-rounds.svg (animated) ════════════════
N = len(STATES)
DUR = 2.5 * N
keyTimes = ";".join("%.4f" % (j / N) for j in range(N + 1))
OX, OY, H, GAP = 178, 96, 30, 46
parts = []
for i, (pair, corpus) in enumerate(STATES):
    made = None if pair is None else pair[0] + pair[1]
    ntok = sum(len(s) for s in corpus)
    vals = ";".join("1" if j == i else "0" for j in range(N)) + ";" + ("1" if i == 0 else "0")
    g = ['<g opacity="%d">' % (1 if i == 0 else 0),
         '  <animate attributeName="opacity" values="%s" keyTimes="%s" '
         'calcMode="discrete" dur="%ss" repeatCount="indefinite"/>' % (vals, keyTimes, DUR)]
    if pair is None:
        g.append(mono(360, 72, "before any merge &#8212; every word is single characters",
                      11, "#5a4220", "middle", "700"))
    else:
        g.append(mono(360, 72, "merge %d: (&#8220;%s&#8221;, &#8220;%s&#8221;) &#8594; &#8220;%s&#8221;"
                      % (i, esc(pair[0]), esc(pair[1]), esc(made)), 11, RED_S, "middle", "700"))
    for r, word in enumerate(WORDS):
        seq = corpus[ROW_OF[r]]
        y = OY + r * GAP
        g.append(mono(OX - 16, y + 20, word, 10, MUTE, "end"))
        x = OX
        for tok in seq:
            box, w = tokbox(x, y, tok, w=max(30, 13 * len(tok) + 16), h=H, hot=(tok == made), fs=13)
            g += box
            x += w + 6
    g.append(mono(360, OY + 4 * GAP + 26, "%d tokens across the whole corpus" % ntok,
                  10, GRN, "middle", "700"))
    g.append("</g>")
    parts.append(NL.join(g))

write("merge-rounds.svg", NL.join([
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 330" width="720" height="330">',
    '  <rect width="720" height="330" fill="%s" rx="12"/>' % BG,
    mono(360, 26, "BPE training, one merge at a time", 11, MUTE, "middle", "600"),
    mono(360, 44, "corpus: low &#215;5, lower &#215;2, newest &#215;3, widest &#215;2 &#8212; four of them "
         "shown. the new token each round is outlined in red.", 8.5, MUTE, "middle"),
    NL.join(parts),
    mono(360, 318, "each merge replaces a pair everywhere at once, so the corpus shortens and the "
         "next round counts different pairs", 8.5, MUTE, "middle"),
    "</svg>", ""]),
    "%d animated frames, tokens %s" % (N, " -> ".join(str(sum(len(s) for s in c)) for _, c in STATES)))

# ════════════════ FIGURE 2 — merge-tree.svg ════════════════
LEAFY, LV = 268, [268, 208, 152, 96]


def tree(leaves, joins, x0, gap=58):
    """leaves: characters. joins: (leftKey, rightKey, mergedToken, level)."""
    g, named = [], {}
    for i, ch in enumerate(leaves):
        b, _ = tokbox(x0 + i * gap, LEAFY, ch, w=34, h=24, fs=13)
        g += b
        named[ch + "#" + str(i)] = (x0 + i * gap + 17, LEAFY)
    for lkey, rkey, made, lvl in joins:
        lx, ly = named[lkey]
        rx, ry = named[rkey]
        cx, cy = (lx + rx) / 2.0, LV[lvl]
        w = max(34, 13 * len(made) + 16)
        for chx, chy in ((lx, ly), (rx, ry)):
            g.append('  <path d="M %s %s L %s %s L %s %s" fill="none" stroke="%s" '
                     'stroke-width="1.4" opacity="0.6"/>' % (chx, chy, chx, cy + 34, cx, cy + 34, BLUE_S))
        g.append('  <line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="1.4" '
                 'opacity="0.65"/>' % (cx, cy + 34, cx, cy + 24, RED_S))
        b, _ = tokbox(cx - w / 2.0, cy, made, w=w, h=24, hot=True, fs=13)
        g += b
        g.append(mono(cx + w / 2.0 + 7, cy + 16, "merge %d" % MERGE_NUMBER[made], 8.5, MUTE))
        named[made + "#"] = (cx, cy)
    return g

svg3 = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 340" width="720" height="340">',
        '  <rect width="720" height="340" fill="%s" rx="12"/>' % BG,
        mono(360, 26, "merges compose &#8212; each new token is built from tokens earlier merges made",
             11, MUTE, "middle", "600"),
        mono(360, 43, "read bottom-up: characters at the base, the token a merge produced above the "
             "pair it consumed", 8.5, MUTE, "middle"),
        '  <line x1="252" y1="62" x2="252" y2="308" stroke="#e0d5c0" stroke-width="1.5"/>']
svg3 += tree(["l", "o", "w"],
             [("l#0", "o#1", "lo", 1), ("lo#", "w#2", "low", 2)], 62)
svg3 += tree(["n", "e", "w", "e", "s", "t"],
             [("e#3", "s#4", "es", 1), ("es#", "t#5", "est", 2),
              ("n#0", "e#1", "ne", 1), ("ne#", "w#2", "new", 2),
              ("new#", "est#", "newest", 3)], 300, gap=56)
svg3.append(mono(360, 328, "&#8220;est&#8221; is a real English suffix. nothing told the algorithm "
                 "about suffixes &#8212; it counted pairs.", 9, MUTE, "middle"))
svg3.append("</svg>")
write("merge-tree.svg", NL.join(svg3) + NL, "two trees, 7 composed nodes")

# ════════════════ FIGURE 3 — order-matters.svg ════════════════
def trace(word, rules):
    """Apply rules one at a time, recording whether each actually fired."""
    seq = list(word)
    steps = [(None, list(seq), False)]
    for rule in rules:
        nxt = merge_pair([seq], rule)[0]
        steps.append((rule, list(nxt), len(nxt) != len(seq)))
        seq = nxt
    return steps


IN_ORDER = trace("lowest", MERGES)
REVERSED = trace("lowest", list(reversed(MERGES)))
ROWH, TOPY = 27, 96


def column(steps, x0, title, sub, subcol):
    g = [mono(x0 + 175, TOPY - 34, title, 11, INK, "middle", "700"),
         mono(x0 + 175, TOPY - 18, sub, 9, subcol, "middle", "600")]
    for i, (rule, seq, fired) in enumerate(steps):
        y = TOPY + i * ROWH
        label = "start" if rule is None else '(&#8220;%s&#8221;,&#8220;%s&#8221;)' % (rule[0], rule[1])
        g.append(mono(x0 + 100, y + 14, label, 9, INK if fired else MUTE, "end",
                      "700" if fired else "400"))
        # a rule that changed nothing gets a dot, so "fired" reads at a glance
        g.append(mono(x0 + 110, y + 14, "&#8594;" if fired else "&#183;", 10,
                      RED_S if fired else "#c9bfa8"))
        x = x0 + 126
        previous = steps[i - 1][1] if i else []
        for tok in seq:
            box, w = tokbox(x, y, tok, hot=fired and len(tok) > 1 and tok not in previous)
            g += box
            x += w + 4
    n = len(steps[-1][1])
    g.append(mono(x0 + 175, TOPY + len(steps) * ROWH + 22, "result: %d tokens" % n,
                  11, GRN if n == 2 else RED_S, "middle", "700"))
    return g


H2 = TOPY + len(IN_ORDER) * ROWH + 62
svg2 = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 %d" width="780" height="%d">' % (H2, H2),
        '  <rect width="780" height="%d" fill="%s" rx="12"/>' % (H2, BG),
        mono(390, 26, "the same eight rules, applied in two different orders", 11, MUTE, "middle", "600"),
        mono(390, 43, "encoding the word &#8220;lowest&#8221; &#8212; a rule marked &#8594; changed the "
             "sequence, a rule marked &#183; found nothing to merge", 8.5, MUTE, "middle"),
        '  <line x1="390" y1="60" x2="390" y2="%d" stroke="#e0d5c0" stroke-width="1.5"/>' % (H2 - 42)]
svg2 += column(IN_ORDER, 20, "training order", "correct", GRN)
svg2 += column(REVERSED, 402, "reversed", "same rules, wrong order", RED_S)
svg2.append(mono(390, H2 - 18, "(&#8220;lo&#8221;,&#8220;w&#8221;) can only fire after "
                 "(&#8220;l&#8221;,&#8220;o&#8221;) has created &#8220;lo&#8221; &#8212; a rule consumes "
                 "what an earlier rule produced", 9, MUTE, "middle"))
svg2.append("</svg>")
write("order-matters.svg", NL.join(svg2) + NL,
      "%d rows/col, %d tokens in order vs %d reversed"
      % (len(IN_ORDER), len(IN_ORDER[-1][1]), len(REVERSED[-1][1])))

print(NL + "merge rules derived: " + "  ".join('("%s","%s")' % r for r in MERGES))
