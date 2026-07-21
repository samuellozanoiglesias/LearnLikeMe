"""
Turns raw MNIST images into the perceptual, image-grounded arithmetic datasets
used by the digit-recognition front end (perceptual_models.DigitCNN).

Suggested location in the repo: little_learner/modules/perceptual_module/utils.py

Three jobs live here:
  1) load + index MNIST by digit class, keeping train/test images strictly
     separate so nothing evaluated on later has been seen during any
     training stage (recognizer, extractors, or decision module).
  2) on-the-fly batch generation for training (mirrors
     extractor_modules.utils.generate_batch_data, but samples an image per
     digit instead of adding scalar noise).
  3) a *fixed, cached* perceptual test set, so every checkpoint you evaluate
     later sees exactly the same images (see generate_mnist_test_pairs.py).
"""
import os
import numpy as np
import jax.numpy as jnp
from jax import random as jrandom


def load_mnist():
    """
    Loads MNIST train/test images + labels, normalized to [0, 1] float32,
    shape (N, 28, 28, 1).

    Uses tf.keras's bundled loader for convenience. If you'd rather not pull in
    TensorFlow just for this, swap this function's body for
    torchvision.datasets.MNIST or tensorflow_datasets -- nothing downstream
    depends on how the arrays were produced.
    """
    from tensorflow.keras.datasets import mnist  # local import: keep TF optional
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) / 255.0)[..., None]
    x_test = (x_test.astype(np.float32) / 255.0)[..., None]
    return (x_train, y_train.astype(np.int32)), (x_test, y_test.astype(np.int32))


def index_by_digit(images: np.ndarray, labels: np.ndarray) -> dict:
    """Groups images into {digit: array_of_images} for fast per-class sampling."""
    return {d: images[labels == d] for d in range(10)}


def sample_images_for_digits(digit_values, pools: dict, rng: np.random.Generator):
    """
    digit_values: 1-D int array/list, shape (batch,) -- the digit identities needed.
    pools: {digit: array_of_images}, from index_by_digit (pass the TRAIN pools during
           training, the TEST pools during evaluation -- never mix the two).
    Returns: array of shape (batch, 28, 28, 1)
    """
    imgs = []
    for d in np.asarray(digit_values).astype(int):
        pool = pools[int(d)]
        idx = rng.integers(0, len(pool))
        imgs.append(pool[idx])
    return np.stack(imgs, axis=0)


def digits_matrix(values, n_digits: int) -> np.ndarray:
    """Decomposes an int array into base-10 digits, most-significant digit first."""
    values = np.asarray(values)
    out = np.zeros((len(values), n_digits), dtype=int)
    for k in range(n_digits):
        out[:, n_digits - 1 - k] = (values // (10 ** k)) % 10
    return out


# --------------------------------------------------------------------------- #
# Training-time batch generation (single-digit pairs, for the extractor modules)
# --------------------------------------------------------------------------- #

def generate_mnist_batch(pairs, batch_size: int, pools: dict, module_name: str, seed: int,
                          distribution: str = "none", alpha: float = 0.1):
    """
    Perceptual analogue of extractor_modules.utils.generate_batch_data.
    Samples `batch_size` single-digit (a, b) pairs -- using the same
    curriculum-learning distribution options (`distribution`, `alpha`) as
    generate_batch_data -- and attaches one MNIST image per digit, drawn from
    `pools` (use TRAIN pools here).
 
    Args:
        pairs: List of all possible (a, b) pairs to sample from
        batch_size: Number of samples in the batch
        pools: dict mapping digit -> pool of MNIST images for that digit
        module_name: 'carry_extractor' or 'unit_extractor'
        seed: Random seed
        distribution: 'decreasing_exponential', 'balanced', or anything else for uniform
        alpha: Alpha parameter for exponential decay (only used if distribution == 'decreasing_exponential')
 
    Returns:
        images_a, images_b: (batch, 28, 28, 1) jnp arrays
        y: (batch,) jnp int array -- units digit of a+b, or the carry flag,
           depending on module_name.
    """
    if distribution == "decreasing_exponential":
        # Curriculum learning with exponential decay probabilities -- easy
        # (small-sum) pairs are seen far more often early in training.
        probabilities = jnp.array([jnp.exp(-alpha * (a + b)) for a, b in pairs])
        probabilities = probabilities / jnp.sum(probabilities)
 
        pair_rng = jrandom.PRNGKey(seed)
        indices = jrandom.choice(pair_rng, len(pairs), shape=(batch_size,), p=probabilities)
        batch_pairs = [pairs[i] for i in indices]
 
    elif distribution == "balanced":
        # Categorize by sum difficulty, sample equally from each category.
        small_pairs = [(a, b) for a, b in pairs if (a + b) < 7]
        medium_pairs = [(a, b) for a, b in pairs if 7 <= (a + b) <= 12]
        large_pairs = [(a, b) for a, b in pairs if (a + b) > 12]
 
        pair_rng = jrandom.PRNGKey(seed)
        keys = jrandom.split(pair_rng, 3)
 
        samples_per_category = batch_size // 3
        remaining = batch_size % 3
 
        small_indices = jrandom.choice(keys[0], len(small_pairs), shape=(samples_per_category,))
        medium_indices = jrandom.choice(keys[1], len(medium_pairs), shape=(samples_per_category,))
        large_indices = jrandom.choice(keys[2], len(large_pairs), shape=(samples_per_category,))
 
        batch_pairs = ([small_pairs[i] for i in small_indices] +
                       [medium_pairs[i] for i in medium_indices] +
                       [large_pairs[i] for i in large_indices])
 
        if remaining > 0:
            extra_key = jrandom.split(keys[0], 1)[0]
            extra_indices = jrandom.choice(extra_key, len(pairs), shape=(remaining,))
            batch_pairs.extend([pairs[i] for i in extra_indices])
 
    else:
        # Default: uniform sampling -- identical to the original implementation.
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(pairs), size=batch_size)
        batch_pairs = [pairs[i] for i in idx]
 
    a_vals = np.array([p[0] for p in batch_pairs])
    b_vals = np.array([p[1] for p in batch_pairs])
 
    image_rng = np.random.default_rng(seed + 1000)
    images_a = sample_images_for_digits(a_vals, pools, image_rng)
    images_b = sample_images_for_digits(b_vals, pools, image_rng)
 
    if module_name == "unit_extractor":
        y = (a_vals + b_vals) % 10
    elif module_name == "carry_extractor":
        y = (a_vals + b_vals >= 10).astype(int)
    else:
        raise ValueError("module_name must be 'unit_extractor' or 'carry_extractor'")
 
    return jnp.array(images_a), jnp.array(images_b), jnp.array(y)


# --------------------------------------------------------------------------- #
# Fixed, cached perceptual test set
# --------------------------------------------------------------------------- #

def build_and_cache_test_pairs_with_images(test_pairs, pools_test: dict, save_path: str,
                                            number_size: int, n_repeats: int = 1, seed: int = 12345):
    """
    Recreates a *fixed*, reproducible perceptual version of an existing symbolic
    test set (e.g. datasets/2-digit/stimuli_test_pairs.txt) by attaching MNIST
    TEST-split images (held out from every training stage) to every digit
    position of every pair, and caching the result to disk so every checkpoint
    you evaluate later sees the exact same images -- otherwise re-sampling a
    fresh image each evaluation call would make checkpoints incomparable.

    test_pairs: list of (a, b) tuples (multi-digit numbers, symbolic pipeline format).
    pools_test: {digit: array_of_images}, built from index_by_digit on the MNIST
                TEST split ONLY -- never the train split.
    n_repeats: draw this many independent images per digit per pair, so you can
               report mean +/- std accuracy over perceptual draws instead of a
               single (possibly lucky/unlucky) sample.
    """
    rng = np.random.default_rng(seed)
    a_vals = np.array([p[0] for p in test_pairs])
    b_vals = np.array([p[1] for p in test_pairs])
    a_digits = digits_matrix(a_vals, number_size)  # (N, number_size), MSD first
    b_digits = digits_matrix(b_vals, number_size)

    all_a_images, all_b_images = [], []
    for _ in range(n_repeats):
        a_imgs = np.stack(
            [sample_images_for_digits(a_digits[:, k], pools_test, rng) for k in range(number_size)],
            axis=1,
        )  # (N, number_size, 28, 28, 1)
        b_imgs = np.stack(
            [sample_images_for_digits(b_digits[:, k], pools_test, rng) for k in range(number_size)],
            axis=1,
        )
        all_a_images.append(a_imgs)
        all_b_images.append(b_imgs)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        pairs=np.array(test_pairs),
        a_images=np.stack(all_a_images, axis=0),  # (n_repeats, N, number_size, 28, 28, 1)
        b_images=np.stack(all_b_images, axis=0),
        n_repeats=n_repeats,
        seed=seed,
    )
    print(f"Saved perceptual test set ({len(test_pairs)} pairs x {n_repeats} repeat(s)) to {save_path}")


def load_cached_test_pairs_with_images(save_path: str):
    """Loads a perceptual test set produced by build_and_cache_test_pairs_with_images."""
    data = np.load(save_path, allow_pickle=False)
    return {
        "pairs": data["pairs"],
        "a_images": data["a_images"],
        "b_images": data["b_images"],
        "n_repeats": int(data["n_repeats"]),
        "seed": int(data["seed"]),
    }
