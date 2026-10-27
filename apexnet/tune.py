"""Hyper-parameter search: random search or successive halving (Hyperband's inner loop), no extra dependencies.

    space = {"lr": an.tune.LogUniform(1e-4, 1e-1), "width": an.tune.Choice([32, 64, 128]), "depth": an.tune.Int(1, 3)}
    def build(p): return an.Trainer(an.models.MLP([p["width"]] * p["depth"], 3), an.cross_entropy(), an.AdamW(p["lr"]), ...)
    best = an.tune.search(build, space, train_ds, val_data=val_ds, metric="val_loss", mode="min",
                          trials=12, epochs=9, method="halving")
"""
import math
import numpy as np


class Choice:
    def __init__(self, options):
        self.options = list(options)

    def sample(self, rs):
        return self.options[rs.randint(len(self.options))]


class Uniform:
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def sample(self, rs):
        return float(rs.uniform(self.lo, self.hi))


class LogUniform(Uniform):
    def sample(self, rs):
        return float(math.exp(rs.uniform(math.log(self.lo), math.log(self.hi))))


class Int:
    """Inclusive integer range."""
    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def sample(self, rs):
        return int(rs.randint(self.lo, self.hi + 1))


def _draw(space, rs):
    return {k: (v.sample(rs) if hasattr(v, "sample") else v) for k, v in space.items()}


def search(build, space, data, y=None, *, val_data=None, metric="val_loss", mode="min", trials=8, epochs=6,
           method="random", eta=3, seed=0, fit_kwargs=None, verbose=1):
    """``build(params) -> Trainer`` (fresh model each call). Returns a dict with ``best_params``, ``best_score``,
    ``trials`` (list of {params, score, epochs}) and ``best_trainer`` (the winning, trained Trainer)."""
    rs = np.random.RandomState(seed)
    sign = 1.0 if mode == "min" else -1.0
    kw = dict(fit_kwargs or {})
    kw.setdefault("verbose", 0)
    if val_data is not None:
        kw["val_data"] = val_data
    log = []

    def score_of(trainer):
        h = trainer.history
        return float(h[metric][-1]) if metric in h else float("inf") * sign

    def say(*a):
        if verbose:
            print(*a, flush=True)

    live = []
    for i in range(trials):
        p = _draw(space, rs)
        live.append({"params": p, "trainer": build(p), "epochs": 0, "score": None, "id": i})
    budget = max(1, epochs if method == "random" else max(1, epochs // (eta ** max(1, int(math.log(trials, eta))))))
    rung = 0
    while True:
        for t in live:
            t["trainer"].fit(data, y, epochs=t["epochs"] + budget, **kw)
            t["epochs"] += budget
            t["score"] = score_of(t["trainer"])
            say(f"  trial {t['id']:>2} epochs={t['epochs']:<3} {metric}={t['score']:.4g} {t['params']}")
        for t in live:
            log.append({"params": t["params"], "score": t["score"], "epochs": t["epochs"]})
        if method == "random" or len(live) == 1 or t["epochs"] >= epochs:
            break
        live.sort(key=lambda t: sign * t["score"])
        live = live[: max(1, len(live) // eta)]
        budget = min(budget * eta, epochs - live[0]["epochs"])
        if budget <= 0:
            break
        rung += 1
        say(f" -- rung {rung}: keeping {len(live)} trial(s), +{budget} epochs")
    best = min(live, key=lambda t: sign * t["score"])
    return {"best_params": best["params"], "best_score": best["score"], "trials": log, "best_trainer": best["trainer"]}
