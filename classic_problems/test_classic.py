"""
Unit tests for classic_problems/fizzbuzz.py

Run:  python -m pytest classic_problems/ -v
 or:  python classic_problems/test_classic.py     (no pytest required)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fizzbuzz import (anagrams, fibonacci, fizzbuzz, fizzbuzz_lazy, fizzbuzz_naive,
                      is_palindrome, is_prime, reverse_words, two_sum)

EXPECTED_15 = ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
               "11", "Fizz", "13", "14", "FizzBuzz"]


# ---------------------------------------------------------------- FizzBuzz
def test_fizzbuzz_first_fifteen():
    assert fizzbuzz(15) == EXPECTED_15


def test_naive_matches_rule_driven():
    assert fizzbuzz_naive(100) == fizzbuzz(100)


def test_fizzbuzz_zero_is_empty():
    assert fizzbuzz(0) == []


def test_fizzbuzz_rejects_negative():
    try:
        fizzbuzz(-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_fizzbuzz_multiples():
    result = fizzbuzz(30)
    assert result[2] == "Fizz"        # 3
    assert result[4] == "Buzz"        # 5
    assert result[14] == "FizzBuzz"   # 15
    assert result[29] == "FizzBuzz"   # 30


def test_fizzbuzz_custom_rules():
    rules = ((3, "Fizz"), (5, "Buzz"), (7, "Bazz"))
    result = fizzbuzz(105, rules)
    assert result[6] == "Bazz"            # 7
    assert result[20] == "FizzBazz"       # 21 = 3 * 7
    assert result[34] == "BuzzBazz"       # 35 = 5 * 7
    assert result[104] == "FizzBuzzBazz"  # 105 = 3 * 5 * 7


def test_lazy_matches_eager():
    gen = fizzbuzz_lazy()
    assert [next(gen) for _ in range(15)] == EXPECTED_15


# ---------------------------------------------------------------- others
def test_fibonacci():
    assert fibonacci(10) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    assert fibonacci(0) == []
    assert fibonacci(1) == [0]


def test_primes():
    assert [x for x in range(30) if is_prime(x)] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert not is_prime(1)
    assert not is_prime(0)
    assert not is_prime(-7)
    assert is_prime(7919)


def test_palindrome():
    assert is_palindrome("A man, a plan, a canal: Panama")
    assert is_palindrome("")
    assert not is_palindrome("hello")


def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == (0, 1)
    assert two_sum([3, 2, 4], 6) == (1, 2)
    assert two_sum([1, 2, 3], 100) is None


def test_reverse_words():
    assert reverse_words("  the sky   is blue ") == "blue is sky the"
    assert reverse_words("") == ""


def test_anagrams():
    assert anagrams("Listen", "Silent")
    assert anagrams("Dormitory", "Dirty Room")
    assert not anagrams("hello", "world")


def _run_standalone() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    print("Running classic-problem tests (standalone, no pytest)\n")
    raise SystemExit(_run_standalone())
