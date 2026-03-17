"""A deliberately verbose Python script for optimization.

The goal: reduce character count while preserving all functionality.
The eval metric is `wc -c` (minimize). The agent should refactor
for conciseness without breaking the output.
"""

import sys
from typing import Optional


def calculate_fibonacci_sequence(number_of_terms: int) -> list[int]:
    """Calculate the Fibonacci sequence up to n terms."""
    if number_of_terms <= 0:
        return []
    if number_of_terms == 1:
        return [0]

    fibonacci_numbers: list[int] = [0, 1]
    for index in range(2, number_of_terms):
        next_fibonacci_number = fibonacci_numbers[index - 1] + fibonacci_numbers[index - 2]
        fibonacci_numbers.append(next_fibonacci_number)

    return fibonacci_numbers


def check_if_number_is_prime(number_to_check: int) -> bool:
    """Check if a given number is a prime number."""
    if number_to_check < 2:
        return False
    for potential_divisor in range(2, int(number_to_check**0.5) + 1):
        if number_to_check % potential_divisor == 0:
            return False
    return True


def find_all_prime_numbers_in_range(
    start_of_range: int, end_of_range: int
) -> list[int]:
    """Find all prime numbers within a given range."""
    prime_numbers_found: list[int] = []
    for current_number in range(start_of_range, end_of_range + 1):
        if check_if_number_is_prime(current_number):
            prime_numbers_found.append(current_number)
    return prime_numbers_found


def calculate_average_of_numbers(list_of_numbers: list[float]) -> Optional[float]:
    """Calculate the arithmetic mean of a list of numbers."""
    if len(list_of_numbers) == 0:
        return None
    total_sum_of_all_numbers = 0.0
    for individual_number in list_of_numbers:
        total_sum_of_all_numbers += individual_number
    arithmetic_mean = total_sum_of_all_numbers / len(list_of_numbers)
    return arithmetic_mean


def reverse_a_string_character_by_character(input_string: str) -> str:
    """Reverse a string by iterating through each character."""
    reversed_string_result = ""
    for character_index in range(len(input_string) - 1, -1, -1):
        reversed_string_result += input_string[character_index]
    return reversed_string_result


def count_occurrences_of_each_word_in_text(text_content: str) -> dict[str, int]:
    """Count how many times each word appears in the given text."""
    word_count_dictionary: dict[str, int] = {}
    list_of_words = text_content.lower().split()
    for individual_word in list_of_words:
        if individual_word in word_count_dictionary:
            word_count_dictionary[individual_word] += 1
        else:
            word_count_dictionary[individual_word] = 1
    return word_count_dictionary


def main() -> None:
    """Run all functions and print results."""
    fibonacci_result = calculate_fibonacci_sequence(10)
    print(f"Fibonacci(10): {fibonacci_result}")

    primes_result = find_all_prime_numbers_in_range(1, 50)
    print(f"Primes(1-50): {primes_result}")

    numbers_for_average = [10.0, 20.0, 30.0, 40.0, 50.0]
    average_result = calculate_average_of_numbers(numbers_for_average)
    print(f"Average: {average_result}")

    original_string = "Hello, World!"
    reversed_result = reverse_a_string_character_by_character(original_string)
    print(f"Reverse: {reversed_result}")

    sample_text = "the quick brown fox jumps over the lazy dog the fox"
    word_counts = count_occurrences_of_each_word_in_text(sample_text)
    print(f"Word counts: {word_counts}")


if __name__ == "__main__":
    main()
