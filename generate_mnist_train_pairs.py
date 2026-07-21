# USE: nohup python generate_mnist_train_pairs.py cuenca 2 5 54321 &> generate_mnist_train_pairs.log &
#
# Training-set counterpart to generate_mnist_test_pairs.py: same mechanism,
# same cached-npz idea, but for train_pairs_not_in_stimuli.txt instead of
# stimuli_test_pairs.txt, and drawing images from the MNIST TRAIN split
# instead of the TEST split (the TEST split must stay reserved exclusively
# for held-out evaluation -- see generate_mnist_test_pairs.py's docstring).
#
# This reuses build_and_cache_test_pairs_with_images() UNCHANGED -- it's a
# generic (pairs, pools) -> cached-images-npz function despite the "test" in
# its name/parameter names; nothing about it is test-specific. Passing our
# training pairs/pools through it means the cache is guaranteed to have
# exactly the same on-disk format as mnist_stimuli_test_pairs.npz.
#
# NOTE on how this cache is (and isn't) used downstream: train_decision_
# module_perceptual.py and train_all_at_once_perceptual.py do NOT read this
# npz back in by default. They draw MNIST TRAIN-split images on the fly, a
# fresh random draw every batch, the same way train_extractor_modules_
# perceptual.py already does -- because for TRAINING (unlike the fixed TEST
# set) fresh draws are arguably the more correct choice: they expose the
# model to many different handwritten instances of each digit over the
# course of training instead of memorizing N_REPEATS fixed ones, which acts
# as a natural form of data augmentation/regularization. This generator is
# still provided because you asked for parity with the test-set pipeline --
# it's a correct, reusable artifact (e.g. for reproducible small-scale
# ablations, debugging, or visualization) -- but treat N_REPEATS here as a
# deliberate diversity/reproducibility trade-off, not a drop-in requirement
# for the training scripts to run.
import os
import sys

import numpy as np

from little_learner.modules.perceptual_module.mnist_utils import load_mnist, index_by_digit, build_and_cache_test_pairs_with_images

# --- Config ---
CLUSTER = str(sys.argv[1]).lower()
NUMBER_SIZE = int(sys.argv[2])
N_REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 5  # independent image draws per digit
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 54321

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
TRAIN_PAIRS_PATH = os.path.join(DATASET_DIR, "train_pairs_not_in_stimuli.txt")
SAVE_PATH = os.path.join(DATASET_DIR, "mnist_train_pairs.npz")


def load_pairs(path):
    with open(path, "r") as f:
        content = f.read().strip()
        return eval(content) if content else []


train_pairs = load_pairs(TRAIN_PAIRS_PATH)
if not train_pairs:
    raise RuntimeError(f"No pairs found in {TRAIN_PAIRS_PATH}. Run generate_arithmetic_datasets.py first "
                        "-- this script re-images an existing symbolic training set, it doesn't invent one.")
print(f"Loaded {len(train_pairs)} symbolic training pairs from {TRAIN_PAIRS_PATH}")

# MNIST TRAIN split only -- these images are fine to reuse across many
# training steps/epochs (unlike the TEST split, which must never appear in
# any training stage).
(x_train, y_train), _ = load_mnist()
pools_train = index_by_digit(x_train, y_train)
print("MNIST train-split pool sizes per digit:", {d: len(pools_train[d]) for d in range(10)})

build_and_cache_test_pairs_with_images(
    test_pairs=train_pairs,
    pools_test=pools_train,
    save_path=SAVE_PATH,
    number_size=NUMBER_SIZE,
    n_repeats=N_REPEATS,
    seed=SEED,
)

# --- Same hard guarantee generate_mnist_test_pairs.py makes: reload the
# cache and check the pairs are byte-for-byte identical (same additions,
# same order) to the symbolic training set we started from.
_check = np.load(SAVE_PATH, allow_pickle=False)
cached_pairs = [tuple(p) for p in _check["pairs"].tolist()]
assert cached_pairs == train_pairs, (
    "Cached MNIST training pairs do not match datasets/{N}-digit/train_pairs_not_in_stimuli.txt exactly. "
    "Do not use this cache until this is fixed."
)
print(f"Verified: {len(cached_pairs)} cached pairs match train_pairs_not_in_stimuli.txt exactly, "
      f"in the same order, with {2 * NUMBER_SIZE} MNIST train-split images attached per pair "
      f"({N_REPEATS} independent draw(s) each).")
