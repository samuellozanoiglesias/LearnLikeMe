import os
import random
from typing import List, Tuple, Set

class ArithmeticDatasetGenerator:
    def __init__(self, output_dir: str = "datasets"):
        """
        Initialize the ArithmeticDatasetGenerator with the output directory.
        
        Args:
            output_dir (str): Directory where the generated datasets will be saved
        """
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
        """Generate all valid two-digit addition pairs where sum < 100."""
        data = [
            (a, b) for a in range(100) for b in range(100)
            if a + b < 100
        ]
        self._save_dataset(data, "all_valid_additions.txt")
        return data

    def generate_carry_operations(self) -> List[Tuple[int, int]]:
        """Generate additions that involve carry operations."""
        data = []
        for a in range(100):
            for b in range(100):
                units_a, tens_a = a % 10, a // 10
                units_b, tens_b = b % 10, b // 10
                
                has_carry = (units_a + units_b >= 10) or (tens_a + tens_b >= 10)
                if has_carry and (a + b < 100):
                    data.append((a, b))
        
        self._save_dataset(data, "carry_additions.txt")
        return data

    def generate_problem_size_datasets(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Generate datasets separated by problem size (small: sum < 40, large: 60 < sum < 100)."""
        small_problems = [
            (a, b) for a in range(100) for b in range(100)
            if a + b < 40
        ]
        self._save_dataset(small_problems, "small_additions.txt")

        large_problems = [
            (a, b) for a in range(100) for b in range(100)
            if 60 < a + b < 100
        ]
        self._save_dataset(large_problems, "large_additions.txt")
        
        return small_problems, large_problems

    def generate_stimuli_pairs(self) -> List[Tuple[int, int]]:
        """Generate specific test pairs for stimuli evaluation."""
        test_pairs_stimuli = [
            (4, 3), (3, 4), (25, 62), (62, 25), (6, 13), (13, 6), (37, 41), (41, 37),
            (7, 2), (2, 7), (24, 62), (62, 24), (12, 7), (7, 12), (42, 34), (34, 42),
            (5, 14), (14, 5), (21, 74), (74, 21), (12, 13), (13, 12), (24, 45), (45, 24),
            (16, 12), (12, 16), (24, 71), (71, 24), (4, 13), (13, 4), (41, 35), (35, 41),
            (14, 15), (15, 14), (65, 32), (32, 65), (13, 16), (16, 13), (43, 25), (25, 43),
            (5, 3), (3, 5), (65, 21), (21, 65), (12, 15), (15, 12), (42, 32), (32, 42),
            (14, 12), (12, 14), (41, 38), (38, 41), (15, 13), (13, 15), (74, 23), (23, 74),
            (13, 14), (14, 13), (31, 45), (45, 31), (4, 15), (15, 4), (65, 31), (31, 65),
            (2, 13), (13, 2), (24, 43), (43, 24), (2, 17), (17, 2), (61, 32), (32, 61),
            (14, 4), (4, 14), (32, 47), (47, 32), (12, 5), (5, 12), (53, 36), (36, 53),
            (16, 3), (3, 16), (28, 51), (51, 28), (5, 13), (13, 5), (36, 43), (43, 36),
            (14, 3), (3, 14), (67, 32), (32, 67), (17, 12), (12, 17), (26, 43), (43, 26),
            (5, 7), (7, 5), (47, 38), (38, 47), (13, 8), (8, 13), (26, 65), (65, 26),
            (8, 6), (6, 8), (29, 48), (48, 29), (9, 4), (4, 9), (39, 26), (26, 39),
            (8, 7), (7, 8), (37, 46), (46, 37), (17, 6), (6, 17), (46, 35), (35, 46),
            (7, 17), (17, 7), (34, 57), (57, 34), (19, 5), (5, 19), (36, 47), (47, 36),
            (6, 15), (15, 6), (52, 29), (29, 52), (4, 17), (17, 4), (34, 49), (49, 34),
            (7, 6), (6, 7), (27, 45), (45, 27), (13, 18), (18, 13), (38, 25), (25, 38),
            (9, 5), (5, 9), (25, 67), (67, 25), (5, 16), (16, 5), (65, 28), (28, 65),
            (5, 8), (8, 5), (35, 27), (27, 35), (7, 9), (9, 7), (45, 39), (39, 45),
            (18, 7), (7, 18), (46, 36), (36, 46), (17, 19), (19, 17), (29, 67), (67, 29),
            (14, 17), (17, 14), (39, 28), (28, 39), (19, 15), (15, 19), (49, 32), (32, 49),
            (15, 8), (8, 15), (64, 28), (28, 64), (8, 9), (9, 8), (24, 68), (68, 24),
            (9, 14), (14, 9), (29, 38), (38, 29), (7, 4), (4, 7), (28, 69), (69, 28)
        ]
        self._save_dataset(test_pairs_stimuli, "stimuli_test_pairs.txt")
        return test_pairs_stimuli

    def generate_training_stimuli(self) -> List[Tuple[int, int]]:
        """Generate training stimuli by filtering out test pairs."""
        test_pairs = set(self.generate_stimuli_pairs())
        
        # Generate all possible combinations that are not in test pairs
        train_couples_stimuli = [
            (a, b) for a in range(100) for b in range(100)
            if (a, b) not in test_pairs and a + b < 100
        ]
        
        # Sort by sum for curriculum learning
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
    generator = ArithmeticDatasetGenerator()
    
    # Generate all datasets
    print("\nGenerating basic datasets...")
    generator.generate_all_valid_additions()
    generator.generate_carry_operations()
    generator.generate_problem_size_datasets()
    
    print("\nGenerating stimuli datasets...")
    generator.generate_stimuli_pairs()
    generator.generate_training_stimuli()
    
    print("\nGenerating balanced training sets...")
    generator.generate_balanced_training_set()
    
    print("\nAll datasets have been generated successfully!")

if __name__ == "__main__":
    main()
