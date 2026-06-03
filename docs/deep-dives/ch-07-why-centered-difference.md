# Deep Dive: Why Centered Differences Are O(h²) — and Why `h = 1e-5`

> Optional reading for Chapter 07. No new code required.
> After this you'll know *why* the centered difference is more accurate, and why
> there's a sweet-spot step size instead of "smaller `h` is always better."

---

## The two claims from the chapter

1. The **centered** difference `(f(x+h) − f(x−h)) / 2h` has error proportional to `h²`,
   while the **one-sided** `(f(x+h) − f(x)) / h` has error proportional to `h`.
2. The best `h` is around `1e-5` — *not* the smallest number you can type.

Both fall out of one tool: the Taylor series.

---

## The tool: Taylor expansion

For a smooth function, nudging the input by `h` gives an exact infinite series:

$$
f(x+h) = f(x) + h\,f'(x) + \tfrac{h^2}{2}f''(x) + \tfrac{h^3}{6}f'''(x) + \cdots
$$

Stepping the other way just flips the sign of every *odd* power of `h`:

$$
f(x-h) = f(x) - h\,f'(x) + \tfrac{h^2}{2}f''(x) - \tfrac{h^3}{6}f'''(x) + \cdots
$$

That sign-flip is the whole secret. Watch what each formula does with it.

---

## Part 1: the one-sided difference loses a whole power of h

Subtract `f(x)` and divide by `h`:

$$
\frac{f(x+h) - f(x)}{h}
= f'(x) + \underbrace{\tfrac{h}{2}f''(x)}_{\text{leading error}} + \tfrac{h^2}{6}f'''(x) + \cdots
$$

The first leftover term is `(h/2)·f''(x)` — proportional to **`h`**. So halving `h` only halves the error. This is **first-order accuracy, O(h)**.

---

## Part 2: the centered difference cancels that term

Now subtract the two expansions. Every *even*-power term (the `f`, the `f''`, …) is identical in both, so it cancels; every *odd*-power term doubles:

$$
f(x+h) - f(x-h) = 2h\,f'(x) + \tfrac{2h^3}{6}f'''(x) + \cdots
$$

Divide by `2h`:

$$
\frac{f(x+h) - f(x-h)}{2h}
= f'(x) + \underbrace{\tfrac{h^2}{6}f'''(x)}_{\text{leading error}} + \cdots
$$

The `f''` term — the thing that hurt the one-sided formula — **vanished**. The first surviving error is proportional to **`h²`**. So halving `h` cuts the error by **4×**. This is **second-order accuracy, O(h²)** — same number of function calls, dramatically better.

> **Intuitive version:** the one-sided difference measures the slope of a line that leans to one side of `x`. The centered difference straddles `x` symmetrically, so the curvature it over-counts on the right is exactly cancelled by what it under-counts on the left.

---

## Part 3: so why not take `h = 1e-300`?

If the truncation error is `~h²`, smaller `h` looks strictly better — on paper. But the computer doesn't store `f(x±h)` exactly; each value carries a tiny round-off of about machine epsilon, `ε ≈ 2.2e-16` relative size (Ch 06's floating-point reality).

When `h` is tiny, `f(x+h)` and `f(x−h)` are *nearly equal*, and subtracting two nearly-equal numbers throws away most of the significant digits (**catastrophic cancellation**). That round-off error grows like `ε / h`. So the total error has two competing parts:

$$
\text{error}(h) \;\approx\; \underbrace{\tfrac{h^2}{6}\,|f'''(x)|}_{\text{truncation, grows with }h}
\;+\; \underbrace{\frac{\varepsilon}{h}}_{\text{round-off, grows as }h\to 0}
$$

Make `h` too big and the first term dominates; too small and the second explodes. Minimizing the sum (set the derivative to zero) gives an optimal `h` on the order of the **cube root of machine epsilon**:

$$
h_{\text{opt}} \sim \varepsilon^{1/3} \approx (2.2\times10^{-16})^{1/3} \approx 6\times10^{-6}
$$

which rounds to the rule of thumb the chapter uses: **`h = 1e-5`**. (For the *one-sided* formula the balance lands at `√ε ≈ 1e-8` — different sweet spot, worse accuracy.)

---

## See it happen

```bash
bun -e '
const f = (x) => Math.sin(x);     // f(1) derivative is cos(1) = 0.5403023...
const exact = Math.cos(1);
for (const h of [1e-1, 1e-3, 1e-5, 1e-8, 1e-12]) {
  const centered = (f(1+h) - f(1-h)) / (2*h);
  console.log(`h=${h.toExponential()}  error=${Math.abs(centered-exact).toExponential(2)}`);
}
'
```

You'll see the error shrink as `h` falls from `1e-1` to `1e-5` (truncation improving ~100× per 10× drop in `h`), then *grow* again by `1e-8`–`1e-12` as round-off takes over. The valley sits right around `1e-5` — exactly where theory said it would.

---

## Pen-and-paper exercise

1. Using the two Taylor expansions, derive the centered **second** derivative formula
   `f''(x) ≈ (f(x+h) − 2f(x) + f(x−h)) / h²`. What order is its error?
2. The one-sided formula has optimal `h ≈ √ε ≈ 1e-8`. Plug `h = 1e-8` and `h = 1e-5` into
   the round-off term `ε/h` and the truncation term — confirm why the centered formula
   prefers the *larger* `1e-5`.
3. For `f(x) = x²`, show the centered difference gives the **exact** derivative `2x` for
   *any* `h` (hint: the `f'''` term is zero). Why is a parabola the easy case?
