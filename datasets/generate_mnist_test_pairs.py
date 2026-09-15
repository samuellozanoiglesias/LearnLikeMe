# USE: nohup python generate_mnist_test_pairs.py cuenca 2 3 12345 &> generate_mnist_test_pairs.log &
#
# Answers "how do I recreate my test set with MNIST": it loads the *existing*
# symbolic stimuli_test_pairs.txt (the balanced carry/no-carry x small/large
# set generate_stimuli_test_pairs.py already built), and attaches one or more
# real MNIST images -- drawn only from the MNIST TEST split, never seen during
# any training stage -- to every digit position of every pair. The result is
# cached to disk so every checkpoint you evaluate later sees the exact same
# images; without caching, a fresh random draw each evaluation call would make
# checkpoints incomparable to each other.
import os
import sys

import numpy as np

from little_learner.modules.perceptual_module.mnist_utils import load_mnist, index_by_digit, build_and_cache_test_pairs_with_images

# --- Config ---
CLUSTER = str(sys.argv[1]).lower()
NUMBER_SIZE = int(sys.argv[2])
N_REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # independent image draws per digit
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 12345

if CLUSTER == "cuenca":
    CLUSTER_DIR = ""
    CODE_DIR = "/home/samuel_lozano/LearnLikeMe"
elif CLUSTER == "brigit":
    CLUSTER_DIR = "/mnt/lustre/home/samuloza"
    CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
elif CLUSTER == "local":
    CLUSTER_DIR = "D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
    CODE_DIR = f"{CLUSTER_DIR}/LearnLikeMe"
else:
    raise ValueError("Invalid cluster name. Choose 'cuenca', 'brigit', or 'local'.")

DATASET_DIR = f"{CODE_DIR}/datasets/{NUMBER_SIZE}-digit"
STIMULI_PATH = os.path.join(DATASET_DIR, "stimuli_test_pairs.txt")
SAVE_PATH = os.path.join(DATASET_DIR, "mnist_stimuli_test_pairs.npz")


def load_pairs(path):
    with open(path, "r") as f:
        content = f.read().strip()
        return eval(content) if content else []


test_pairs = load_pairs(STIMULI_PATH)
if not test_pairs:
    raise RuntimeError(f"No pairs found in {STIMULI_PATH}. Run generate_stimuli_test_pairs.py first "
                        "-- this script re-images an existing symbolic test set, it doesn't invent one.")
print(f"Loaded {len(test_pairs)} symbolic test pairs from {STIMULI_PATH}")

# MNIST TEST split only -- these images must never appear anywhere in training
# (not in the recognizer's training set, not in the extractors', not in the
# decision module's). If your recognizer was trained on the full 60k MNIST
# train split, this is automatically satisfied.
_, (x_test, y_test) = load_mnist()
pools_test = index_by_digit(x_test, y_test)
print("MNIST test-split pool sizes per digit:", {d: len(pools_test[d]) for d in range(10)})

build_and_cache_test_pairs_with_images(
    test_pairs=test_pairs,
    pools_test=pools_test,
    save_path=SAVE_PATH,
    number_size=NUMBER_SIZE,
    n_repeats=N_REPEATS,
    seed=SEED,
)

# --- Hard guarantee, not just an assumption: reload the cache and check the
# pairs are byte-for-byte identical (same additions, same order) to the
# symbolic test set we started from. If this ever fails, something upstream
# changed the pairs and the cache should NOT be trusted for evaluation.
_check = np.load(SAVE_PATH, allow_pickle=False)
cached_pairs = [tuple(p) for p in _check["pairs"].tolist()]
assert cached_pairs == test_pairs, (
    "Cached MNIST test pairs do not match datasets/{N}-digit/stimuli_test_pairs.txt exactly. "
    "Do not use this cache for evaluation until this is fixed."
)
print(f"Verified: {len(cached_pairs)} cached pairs match stimuli_test_pairs.txt exactly, "
      f"in the same order, with {2 * NUMBER_SIZE} MNIST test-split images attached per pair "
      f"({N_REPEATS} independent draw(s) each).")