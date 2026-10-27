"""PyTorch and JAX baselines for the benchmark: same architectures / Adam(1e-3) / batch as the ApexNet and Keras runs.
Synthetic repeated batch, so this measures framework + accelerator throughput (not input pipelines).
Each function returns median samples/s over R timed epochs of S steps, after one warm-up epoch."""
import os
import time
import numpy as np


def _median_sps(run_epoch, S, gb, R):
    run_epoch()
    ts = []
    for _ in range(R):
        t0 = time.perf_counter()
        run_epoch()
        ts.append(time.perf_counter() - t0)
    return gb * S / float(np.median(ts))


def _batch(which, c, gb):
    rs = np.random.RandomState(0)
    if which == "mlp":
        d, _, k, _ = c["mlp"]
        return rs.randn(gb, d).astype("float32"), rs.randint(0, k, gb).astype("int64")
    if which == "cnn":
        s, _, k, _ = c["cnn"]
        return rs.randn(gb, s, s, 3).astype("float32"), rs.randint(0, k, gb).astype("int64")
    V, T, *_ = c["tfm"]
    return rs.randint(0, V, (gb, T)).astype("int64"), rs.randint(0, V, (gb, T)).astype("int64")


# ----------------------------------------------------------------------------- PyTorch
def torch_baseline(which, c, gb, S, R, precision="float32"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda" and precision != "float32"
    amp_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    torch.manual_seed(0)
    if which == "mlp":
        d, hid, k, _ = c["mlp"]
        layers, prev = [], d
        for h in hid:
            layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        model = nn.Sequential(*layers, nn.Linear(prev, k))
    elif which == "cnn":
        s, ch, k, _ = c["cnn"]
        layers, prev = [], 3
        for f in ch:
            layers += [nn.Conv2d(prev, f, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]; prev = f
        model = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(prev, k))
    else:
        V, T, dim, depth, heads, _ = c["tfm"]

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
                self.qkv, self.out = nn.Linear(dim, 3 * dim, bias=False), nn.Linear(dim, dim)
                self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))

            def forward(self, x):
                B, T_, D = x.shape
                q, k_, v = self.qkv(self.ln1(x)).view(B, T_, 3, heads, D // heads).permute(2, 0, 3, 1, 4)
                a = F.scaled_dot_product_attention(q, k_, v, is_causal=True).transpose(1, 2).reshape(B, T_, D)
                x = x + self.out(a)
                return x + self.mlp(self.ln2(x))

        class LM(nn.Module):
            def __init__(self):
                super().__init__()
                self.tok, self.pos = nn.Embedding(V, dim), nn.Parameter(torch.randn(T, dim) * 0.02)
                self.blocks, self.ln, self.head = nn.Sequential(*[Block() for _ in range(depth)]), nn.LayerNorm(dim), nn.Linear(dim, V)

            def forward(self, ids):
                return self.head(self.ln(self.blocks(self.tok(ids) + self.pos[: ids.shape[1]])))
        model = LM()
    model.to(dev)
    xn, yn = _batch(which, c, gb)
    x = torch.from_numpy(xn).to(dev)
    if which == "cnn":
        x = x.permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(yn).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and precision == "float16")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev, dtype=amp_dtype, enabled=amp):
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    def epoch():
        for _ in range(S):
            step()
        if dev == "cuda":
            torch.cuda.synchronize()
    return _median_sps(epoch, S, gb, R)


# ----------------------------------------------------------------------------- JAX
def jax_baseline(which, c, gb, S, R, precision="float32"):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # do not grab the GPU memory TF needs
    import jax
    import jax.numpy as jnp
    import optax
    cdt = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.bfloat16}[precision]  # fp16: bf16 stand-in
    key = jax.random.PRNGKey(0)
    ks = iter(jax.random.split(key, 64))
    glorot = lambda shape: jax.random.normal(next(ks), shape, jnp.float32) * np.sqrt(2.0 / sum(shape[-2:]))
    dense = lambda x, p: x @ p["w"].astype(cdt) + p["b"].astype(cdt)
    ln = lambda x, p: (lambda xf: ((xf - xf.mean(-1, keepdims=True)) * jax.lax.rsqrt(xf.var(-1, keepdims=True) + 1e-5) * p["g"] + p["b"]))(x.astype(jnp.float32)).astype(cdt)
    mk = lambda i, o: {"w": glorot((i, o)), "b": jnp.zeros(o)}
    if which == "mlp":
        d, hid, k, _ = c["mlp"]
        dims = [d] + list(hid) + [k]
        params = [mk(a, b) for a, b in zip(dims[:-1], dims[1:])]

        def fwd(p, x):
            x = x.astype(cdt)
            for l in p[:-1]:
                x = jax.nn.relu(dense(x, l))
            return dense(x, p[-1]).astype(jnp.float32)
    elif which == "cnn":
        s, ch, k, _ = c["cnn"]
        prev, convs = 3, []
        for f in ch:
            convs.append({"w": jax.random.normal(next(ks), (3, 3, prev, f), jnp.float32) * np.sqrt(2.0 / (9 * prev)), "b": jnp.zeros(f)}); prev = f
        params = {"convs": convs, "head": mk(prev, k)}

        def fwd(p, x):
            x = x.astype(cdt)
            for l in p["convs"]:
                x = jax.lax.conv_general_dilated(x, l["w"].astype(cdt), (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")) + l["b"].astype(cdt)
                x = jax.lax.reduce_window(jax.nn.relu(x), -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")
            return dense(x.mean((1, 2)), p["head"]).astype(jnp.float32)
    else:
        V, T, dim, depth, heads, _ = c["tfm"]
        lnp = lambda: {"g": jnp.ones(dim), "b": jnp.zeros(dim)}
        params = {"tok": jax.random.normal(next(ks), (V, dim)) * 0.02, "pos": jax.random.normal(next(ks), (T, dim)) * 0.02,
                  "blocks": [{"ln1": lnp(), "ln2": lnp(), "qkv": glorot((dim, 3 * dim)), "out": mk(dim, dim),
                              "m1": mk(dim, 4 * dim), "m2": mk(4 * dim, dim)} for _ in range(depth)], "ln": lnp(), "head": mk(dim, V)}
        mask = jnp.tril(jnp.ones((T, T), bool))

        def block(p, x):
            B = x.shape[0]
            q, k_, v = jnp.split(ln(x, p["ln1"]) @ p["qkv"].astype(cdt), 3, -1)
            sh = lambda t: t.reshape(B, T, heads, dim // heads).transpose(0, 2, 1, 3)
            q, k_, v = sh(q), sh(k_), sh(v)
            att = jnp.where(mask, (q @ k_.transpose(0, 1, 3, 2)).astype(jnp.float32) / np.sqrt(dim // heads), -1e9)
            a = (jax.nn.softmax(att, -1).astype(cdt) @ v).transpose(0, 2, 1, 3).reshape(B, T, dim)
            x = x + dense(a, p["out"])
            return x + dense(jax.nn.gelu(dense(ln(x, p["ln2"]), p["m1"])), p["m2"])

        def fwd(p, ids):
            x = (p["tok"][ids] + p["pos"]).astype(cdt)
            for b in p["blocks"]:
                x = block(b, x)
            return dense(ln(x, p["ln"]), p["head"]).astype(jnp.float32)
    xn, yn = _batch(which, c, gb)
    x, y = jnp.asarray(xn), jnp.asarray(yn)
    opt = optax.adam(1e-3)
    state = opt.init(params)

    @jax.jit
    def step(params, state, x, y):
        def loss_fn(p):
            out = fwd(p, x)
            return optax.softmax_cross_entropy_with_integer_labels(out.reshape(-1, out.shape[-1]), y.reshape(-1)).mean()
        g = jax.grad(loss_fn)(params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state
    holder = {"p": params, "s": state}

    def epoch():
        p, s = holder["p"], holder["s"]
        for _ in range(S):
            p, s = step(p, s, x, y)
        jax.block_until_ready(p)
        holder["p"], holder["s"] = p, s
    return _median_sps(epoch, S, gb, R)
