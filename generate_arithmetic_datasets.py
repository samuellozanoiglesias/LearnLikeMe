# USE: nohup python generate_arithmetic_datasets.py 3 &> generate_arithmetic_datasets.log &

import os
import random
import sys
from typing import List, Tuple, Set

class ArithmeticDatasetGenerator:
    def generate_test_categories(self) -> None:
        """Generate and save four test categories for efficient evaluation."""
        # Load test pairs, carry, small, large
        test_pairs_path = os.path.join(self.output_dir, "stimuli_test_pairs.txt")
        carry_path = os.path.join(self.output_dir, "carry_additions.txt")
        small_path = os.path.join(self.output_dir, "small_additions.txt")
        large_path = os.path.join(self.output_dir, "large_additions.txt")

        def load_set(path):
            with open(path, "r") as f:
                content = f.read().strip()
                return set(eval(content)) if content else set()

        test_pairs = load_set(test_pairs_path)
        carry = load_set(carry_path)
        small = load_set(small_path)
        large = load_set(large_path)

        # Compute categories
        carry_small = sorted(list(test_pairs & carry & small), key=sum)
        carry_large = sorted(list(test_pairs & carry & large), key=sum)
        no_carry_small = sorted(list((test_pairs & small) - carry), key=sum)
        no_carry_large = sorted(list((test_pairs & large) - carry), key=sum)

        # Save each category
        self._save_dataset(carry_small, "test_carry_small.txt")
        self._save_dataset(carry_large, "test_carry_large.txt")
        self._save_dataset(no_carry_small, "test_no_carry_small.txt")
        self._save_dataset(no_carry_large, "test_no_carry_large.txt")
        print("Test categories saved: carry_small, carry_large, no_carry_small, no_carry_large")
    def __init__(self, number_size: int, output_dir: str = "datasets"):
        """
        Initialize the ArithmeticDatasetGenerator with the output directory.
        
        Args:
            number_size (int): Number of digits for the addition problems
            output_dir (str): Directory where the generated datasets will be saved
        """
        output_dir = os.path.join(output_dir, f"{number_size}-digit")
        self.number_size = number_size
        self.max_number = 10 ** number_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def _save_dataset(self, data: List[Tuple[int, int]], filename: str) -> None:
        """
        Save the dataset to a file.
        
        Args:
            data: List of tuples containing number pairs
            filename: Name of the output file
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(str(data))
        print(f'Dataset saved: {filename} with {len(data)} pairs')

    def generate_all_valid_additions(self) -> List[Tuple[int, int]]:
        """Generate all valid N-digit addition pairs where sum < max_number."""
        data = [
            (a, b) for a in range(self.max_number) for b in range(self.max_number)
            if a + b < self.max_number
        ]
        self._save_dataset(data, "all_valid_additions.txt")
        return data

    def generate_carry_operations(self) -> List[Tuple[int, int]]:
        """Generate additions that involve carry operations for N digits."""
        data = []
        for a in range(self.max_number):
            for b in range(self.max_number):
                has_carry = False
                for d in range(self.number_size):
                    digit_a = (a // (10 ** d)) % 10
                    digit_b = (b // (10 ** d)) % 10
                    if digit_a + digit_b >= 10:
                        has_carry = True
                        break
                if has_carry and (a + b < self.max_number):
                    data.append((a, b))
        self._save_dataset(data, "carry_additions.txt")
        return data

    def generate_problem_size_datasets(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Generate datasets separated by problem size for N digits."""
        # Define thresholds as fractions of max_number
        small_threshold = int(0.4 * self.max_number)
        large_min = int(0.6 * self.max_number)
        large_max = self.max_number
        small_problems = [
            (a, b) for a in range(self.max_number) for b in range(self.max_number)
            if a + b < small_threshold
        ]
        self._save_dataset(small_problems, "small_additions.txt")

        large_problems = [
            (a, b) for a in range(self.max_number) for b in range(self.max_number)
            if large_min < a + b < large_max
        ]
        self._save_dataset(large_problems, "large_additions.txt")
        return small_problems, large_problems

    def generate_training_stimuli(self) -> List[Tuple[int, int]]:
        """Generate training stimuli by filtering out test pairs loaded from .txt."""
        # Load test pairs from stimuli_test_pairs.txt
        test_pairs_path = os.path.join(self.output_dir, "stimuli_test_pairs.txt")
        with open(test_pairs_path, "r") as f:
            content = f.read().strip()
            # Expecting a string representation of a list of tuples
            test_pairs = set(eval(content)) if content else set()
        train_couples_stimuli = [
            (a, b) for a in range(self.max_number) for b in range(self.max_number)
            if (a, b) not in test_pairs and a + b < self.max_number
        ]
        train_couples_stimuli.sort(key=lambda pair: sum(pair))
        self._save_dataset(train_couples_stimuli, "train_pairs_not_in_stimuli.txt")
        return train_couples_stimuli

    def generate_balanced_training_set(self, size_per_category: int = 6198) -> List[Tuple[int, int]]:
        """
        Generate a balanced training set with equal representation of different problem types.
        
        Args:
            size_per_category: Number of examples per category
        """
        # Get all our datasets
        all_additions = set(self.generate_all_valid_additions())
        carry_operations = set(self.generate_carry_operations())
        small_problems, large_problems = self.generate_problem_size_datasets()
        small_problems, large_problems = set(small_problems), set(large_problems)

        # Create balanced subsets
        def get_balanced_subset(problems: Set[Tuple[int, int]], size: int) -> List[Tuple[int, int]]:
            return list(problems)[:size]

        # Generate balanced categories
        small_no_carry = get_balanced_subset(
            {p for p in all_additions if p in small_problems and p not in carry_operations},
            size_per_category
        )
        small_with_carry = get_balanced_subset(
            {p for p in all_additions if p in small_problems and p in carry_operations},
            size_per_category
        )
        large_no_carry = get_balanced_subset(
            {p for p in all_additions if p in large_problems and p not in carry_operations},
            size_per_category
        )
        large_with_carry = get_balanced_subset(
            {p for p in all_additions if p in large_problems and p in carry_operations},
            size_per_category
        )

        # Combine all categories
        balanced_dataset = (
            self._repeat_until_size(small_no_carry, size_per_category) +
            self._repeat_until_size(small_with_carry, size_per_category) +
            self._repeat_until_size(large_no_carry, size_per_category) +
            self._repeat_until_size(large_with_carry, size_per_category)
        )

        # Save both shuffled and sorted versions
        shuffled_dataset = balanced_dataset.copy()
        random.shuffle(shuffled_dataset)
        self._save_dataset(shuffled_dataset, "balanced_training_shuffled.txt")
        
        sorted_dataset = sorted(balanced_dataset, key=sum)
        self._save_dataset(sorted_dataset, "balanced_training_sorted.txt")
        
        return balanced_dataset

    def _repeat_until_size(self, data: List[Tuple[int, int]], target_size: int) -> List[Tuple[int, int]]:
        """Repeat elements in the list until it reaches the target size."""
        result = []
        while len(result) < target_size:
            result.extend(data)
        return result[:target_size]

def main():
    """Main function to generate all datasets."""
    if len(sys.argv) < 2:
        print("Usage: python generate_arithmetic_datasets.py <number_size>")
        sys.exit(1)
    number_size = int(sys.argv[1])
    generator = ArithmeticDatasetGenerator(number_size)

    # Generate all datasets
    print(f"\nGenerating basic datasets for N={number_size} digits...")
    generator.generate_all_valid_additions()
    generator.generate_carry_operations()
    generator.generate_problem_size_datasets()

    print("\nGenerating stimuli datasets...")
    generator.generate_training_stimuli()

    print("\nGenerating balanced training sets...")
    generator.generate_balanced_training_set()

    print("\nGenerating test categories for efficient evaluation...")
    generator.generate_test_categories()

    print("\nAll datasets have been generated successfully!")

if __name__ == "__main__":
    main()
