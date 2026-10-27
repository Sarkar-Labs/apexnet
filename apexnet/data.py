"""tf.data helpers."""
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def from_arrays(x, y=None, w=None, *, batch_size=32, shuffle=False, seed=None,
                drop_remainder=False, repeat=False, map_fn=None, prefetch=True):
    """Build a batched pipeline from in-memory arrays (x may be a dict / tuple of arrays)."""
    parts = tuple(t for t in (x, y, w) if t is not None)
    n = tf.nest.flatten(x)[0].shape[0]
    ds = tf.data.Dataset.from_tensor_slices(parts)
    if shuffle:
        ds = ds.shuffle(n, seed=seed, reshuffle_each_iteration=True)
    if map_fn:
        ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(AUTOTUNE) if prefetch else ds


def split(x, y, val_fraction=0.1, seed=0):
    """Random train/val split of numpy arrays -> ((xt, yt), (xv, yv))."""
    import numpy as np
    idx = np.random.RandomState(seed).permutation(len(x))
    k = int(len(x) * val_fraction)
    v, t = idx[:k], idx[k:]
    return (x[t], y[t]), (x[v], y[v])


def lm_windows(tokens, seq_len, batch_size, *, shuffle=True, seed=None, stride=None):
    """Next-token prediction pipeline: yields (tokens[:-1], tokens[1:]) windows."""
    ds = tf.data.Dataset.from_tensor_slices(tokens).window(seq_len + 1, stride or seq_len, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_len + 1, drop_remainder=True))
    if shuffle:
        ds = ds.shuffle(10_000, seed=seed)
    ds = ds.map(lambda w: (w[:-1], w[1:]), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)


# ------------------------------------------------------------------ generators
def from_generator(gen, signature, *, batch_size=32, shuffle_buffer=0, seed=None, drop_remainder=False):
    """``gen``: zero-arg callable yielding elements. ``signature``: tf.TensorSpec structure."""
    ds = tf.data.Dataset.from_generator(gen, output_signature=signature)
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)


# --------------------------------------------------------------------- TFRecord
def _feature(v):
    v = tf.constant(v).numpy() if not hasattr(v, "dtype") else v
    import numpy as np
    v = np.asarray(v)
    if v.dtype.kind in "iub":
        return tf.train.Feature(int64_list=tf.train.Int64List(value=v.reshape(-1).astype("int64")))
    if v.dtype.kind == "f":
        return tf.train.Feature(float_list=tf.train.FloatList(value=v.reshape(-1).astype("float32")))
    vals = [x if isinstance(x, bytes) else str(x).encode() for x in v.reshape(-1)]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=vals))


def write_tfrecord(path, columns):
    """columns: dict name -> array with a leading example axis. Writes one tf.train.Example per row."""
    n = len(next(iter(columns.values())))
    with tf.io.TFRecordWriter(path) as w:
        for i in range(n):
            ex = tf.train.Example(features=tf.train.Features(
                feature={k: _feature(v[i]) for k, v in columns.items()}))
            w.write(ex.SerializeToString())
    return path


def from_tfrecord(files, feature_spec, *, batch_size=32, shuffle_buffer=0, seed=None, map_fn=None,
                  parallel_reads=tf.data.AUTOTUNE, drop_remainder=False):
    """feature_spec e.g. {'x': tf.io.FixedLenFeature([4], tf.float32), 'y': tf.io.FixedLenFeature([], tf.int64)}.
    Yields dicts; use ``map_fn`` to turn them into (x, y)."""
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=parallel_reads)
    if shuffle_buffer:
        ds = ds.shuffle(shuffle_buffer, seed=seed)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).map(
        lambda b: tf.io.parse_example(b, feature_spec), num_parallel_calls=AUTOTUNE)
    if map_fn:
        ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ------------------------------------------------------------------- CSV / images
def from_csv(files, label, *, batch_size=32, shuffle=False, seed=None, as_matrix=True, **kw):
    """Tabular CSV -> (features, label). ``as_matrix`` stacks numeric columns into one float tensor."""
    ds = tf.data.experimental.make_csv_dataset(files, batch_size, label_name=label, shuffle=shuffle,
                                               shuffle_seed=seed, num_epochs=1, **kw)
    if as_matrix:
        ds = ds.map(lambda f, y: (tf.stack([tf.cast(v, tf.float32) for v in f.values()], -1), y))
    return ds.prefetch(AUTOTUNE)


def from_image_dir(root, image_size=(64, 64), *, batch_size=32, shuffle=True, seed=None, channels=3):
    """root/<class_name>/*.png|jpg|... -> (images float32 in [0,1], int labels). Returns (dataset, class_names)."""
    import os
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    files, labels = [], []
    for i, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(root, c))):
            files.append(os.path.join(root, c, f))
            labels.append(i)

    def load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=channels, expand_animations=False)
        img = tf.image.resize(img, image_size)
        return tf.cast(img, tf.float32) / 255.0, y
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(files), seed=seed)
    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE), classes


# ---------------------------------------------------------------- sequences / aug
def bucket_by_length(ds, boundaries, batch_sizes, length_fn=None, pad_value=0, drop_remainder=False):
    """Group variable-length (x, y) elements of similar length; pads within batch (tf.data bucketing)."""
    length_fn = length_fn or (lambda x, *r: tf.shape(x)[0])
    return ds.bucket_by_sequence_length(length_fn, boundaries, batch_sizes, padding_values=pad_value,
                                        drop_remainder=drop_remainder).prefetch(AUTOTUNE)


def augment_images(flip=True, pad_crop=4, brightness=0.0):
    """Per-image tf.data map function for unbatched (image, label) elements."""
    def fn(img, y):
        if flip:
            img = tf.image.random_flip_left_right(img)
        if pad_crop:
            h, w, c = img.shape
            img = tf.image.random_crop(tf.pad(img, [[pad_crop] * 2, [pad_crop] * 2, [0, 0]], mode="REFLECT"), [h, w, c])
        if brightness:
            img = tf.image.random_brightness(img, brightness)
        return img, y
    return fn


def mixup(num_classes, alpha=0.2):
    """Batch-level mixup for (x, int_labels) batches -> (x_mix, soft_labels). Pair with categorical_cross_entropy()."""
    def fn(x, y):
        B = tf.shape(x)[0]
        g1, g2 = tf.random.gamma([], alpha), tf.random.gamma([], alpha)
        lam = g1 / (g1 + g2 + 1e-9)
        idx = tf.random.shuffle(tf.range(B))
        yo = tf.one_hot(y, num_classes)
        return lam * x + (1 - lam) * tf.gather(x, idx), lam * yo + (1 - lam) * tf.gather(yo, idx)
    return fn


def cutmix(num_classes, alpha=1.0):
    """Batch-level CutMix for NHWC images."""
    def fn(x, y):
        B, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        g1, g2 = tf.random.gamma([], alpha), tf.random.gamma([], alpha)
        lam = g1 / (g1 + g2 + 1e-9)
        r = tf.sqrt(1.0 - lam)
        ch, cw = tf.cast(tf.cast(H, tf.float32) * r, tf.int32), tf.cast(tf.cast(W, tf.float32) * r, tf.int32)
        cy, cx = tf.random.uniform([], 0, H, tf.int32), tf.random.uniform([], 0, W, tf.int32)
        y0, y1 = tf.clip_by_value(cy - ch // 2, 0, H), tf.clip_by_value(cy + ch // 2, 0, H)
        x0, x1 = tf.clip_by_value(cx - cw // 2, 0, W), tf.clip_by_value(cx + cw // 2, 0, W)
        rows, cols = tf.range(H)[:, None], tf.range(W)[None, :]
        box = ((rows >= y0) & (rows < y1) & (cols >= x0) & (cols < x1))[None, :, :, None]
        idx = tf.random.shuffle(tf.range(B))
        out = tf.where(box, tf.gather(x, idx), x)
        area = tf.cast((y1 - y0) * (x1 - x0), tf.float32) / tf.cast(H * W, tf.float32)
        yo = tf.one_hot(y, num_classes)
        return out, (1 - area) * yo + area * tf.gather(yo, idx)
    return fn


# ---------------------------------------------------------------- class weights
def class_weight_vector(class_weight, y):
    """Per-example weights (numpy) from {class: weight}; classes not listed get 1.0."""
    import numpy as np
    y = np.asarray(y).astype("int64")
    n = int(max(max(class_weight), int(y.max()))) + 1
    vec = np.ones(n, "float32")
    for k, v in class_weight.items():
        vec[int(k)] = v
    return vec[y]


def with_class_weight(ds, class_weight):
    """Map a dataset of (x, y[, w]) to (x, y, w * class_weight[y]); classes not listed get 1.0."""
    n = int(max(class_weight)) + 1
    vec = [1.0] * n
    for k, v in class_weight.items():
        vec[int(k)] = float(v)
    table = tf.constant(vec, tf.float32)

    def fn(x, y, *w):
        yi = tf.cast(y, tf.int32)
        cw = tf.where(yi < n, tf.gather(table, tf.clip_by_value(yi, 0, n - 1)), 1.0)
        return (x, y, cw * w[0]) if w else (x, y, cw)
    return ds.map(fn, num_parallel_calls=AUTOTUNE)
