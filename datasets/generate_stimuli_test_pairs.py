# USE: nohup python generate_stimuli_test_pairs.py 3 &> generate_stimuli_test_pairs.log &

import random
import sys

def has_carry(a: int, b: int, number_size: int):
    """Return a list of booleans indicating if there is a carry at each digit position (except the last)."""
    carries = []
    carry = 0
    for i in range(number_size):
        digit_a = (a // (10 ** i)) % 10
        digit_b = (b // (10 ** i)) % 10
        s = digit_a + digit_b + carry
        if i < number_size - 1:
            carries.append(s >= 10)
        carry = s // 10
    return carries  # carries[0]=units->tens, carries[1]=tens->hundreds, ...

def generate_pairs(number_size):
    pairs = []
    min_val = 0
    max_val = 10 ** number_size
    for a in range(min_val, max_val):
        for b in range(min_val, max_val):
            total = a + b
            # Avoid trivial and too large sums
            if total == min_val or total >= max_val:
                continue
            carries = has_carry(a, b, number_size)
            pairs.append((a, b, total, carries))
    return pairs

def classify_pairs(pairs, number_size):
    categories = {
        "carry_small": [],
        "carry_large": [],
        "no_carry_small": [],
        "no_carry_large": []
    }
    max_val = 10 ** number_size
    small_thresh = 0.4 * max_val
    large_thresh = 0.6 * max_val
    for a, b, total, carries in pairs:
        has_any_carry = any(carries)
        if total < small_thresh:
            if has_any_carry:
                categories["carry_small"].append((a, b, carries))
            else:
                categories["no_carry_small"].append((a, b, carries))
        elif total > large_thresh:
            if has_any_carry:
                categories["carry_large"].append((a, b, carries))
            else:
                categories["no_carry_large"].append((a, b, carries))
    return categories

def balance_dataset(categories, total_stimuli, seed=0):
    random.seed(seed)
    final = []
    per_category = total_stimuli // 4
    final_counts = {}
    for key in ["carry_small", "carry_large", "no_carry_small", "no_carry_large"]:
        items = [(a, b) for a, b, _ in categories[key]]
        if len(items) < per_category:
            print(f"Warning: not enough items in {key} to sample {per_category}. Using all available.")
            sampled = items
        else:
            sampled = random.sample(items, per_category)
        final.extend(sampled)
        final_counts[key] = len(sampled)
    return final, final_counts

import os

def save_to_txt(pairs, number_size):
    dir_path = os.path.join("datasets", f"{number_size}-digit")
    os.makedirs(dir_path, exist_ok=True)
    filename = os.path.join(dir_path, "stimuli_test_pairs.txt")
    with open(filename, "w") as f:
        f.write("[")
        f.write(",".join(f"({a},{b})" for a, b in pairs))
        f.write("]\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_stimuli_test_pairs.py <number_size>")
        sys.exit(1)
    number_size = int(sys.argv[1])
    N = 10 ** number_size
    total_possible = N * N / 2
    total_stimuli = int(0.005 * total_possible)
    # Make divisible by 4
    total_stimuli -= total_stimuli % 4
    print(f"Generating {total_stimuli} stimuli (0.5% of {total_possible}, divisible by 4, {total_stimuli//4} per category)")
    pairs = generate_pairs(number_size)
    categories = classify_pairs(pairs, number_size)
    print("Category counts before sampling:")
    for key in ["carry_small", "carry_large", "no_carry_small", "no_carry_large"]:
        print(f"  {key}: {len(categories[key])}")
    final_pairs, final_counts = balance_dataset(categories, total_stimuli)
    print("Category counts in final dataset:")
    for key in ["carry_small", "carry_large", "no_carry_small", "no_carry_large"]:
        print(f"  {key}: {final_counts[key]}")
    save_to_txt(final_pairs, number_size)
    print(f"Saved {len(final_pairs)} pairs to datasets/{number_size}-digit/stimuli_test_pairs.txt for number_size={number_size}")
