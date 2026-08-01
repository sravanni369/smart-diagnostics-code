"""
FizzBuzz and other famously viral interview problems, solved in Python.

FizzBuzz became the canonical screening question after Imran Ghory's 2007 blog post
"Why Can't Programmers.. Program?", popularised by Jeff Atwood. The rule:

    for 1..n, print "Fizz" if divisible by 3, "Buzz" if divisible by 5,
    "FizzBuzz" if divisible by both, otherwise the number.

The interesting part is not the puzzle, it is that the naive version is the one people
write and the extensible version is the one they should. Both are here, plus the
relatives that get asked alongside it.

Every function is pure and total: no printing, no globals, no I/O. `main()` does the
printing. That is what makes them testable — see test_classic.py.

Run:  python fizzbuzz.py
"""

from __future__ import annotations

from typing import Iterator, Sequence


# ----------------------------------------------------------------------------------
# FizzBuzz
# ----------------------------------------------------------------------------------

def fizzbuzz_naive(n: int) -> list[str]:
    """The version everybody writes first.

    Correct, but the branch order is load-bearing: swap the first test to last and it
    silently breaks on 15. That fragility is the actual lesson of the exercise.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    out: list[str] = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


DEFAULT_RULES: tuple[tuple[int, str], ...] = ((3, "Fizz"), (5, "Buzz"))


def fizzbuzz(n: int, rules: Sequence[tuple[int, str]] = DEFAULT_RULES) -> list[str]:
    """Rule-driven FizzBuzz — the version worth keeping.

    Adding "Bazz" for 7 is now data, not a new branch, and there is no combinatorial
    explosion of `% 105 == 0` cases.

    >>> fizzbuzz(15)[-1]
    'FizzBuzz'
    >>> fizzbuzz(7, ((3, "Fizz"), (5, "Buzz"), (7, "Bazz")))[-1]
    'Bazz'
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    out: list[str] = []
    for i in range(1, n + 1):
        word = "".join(label for divisor, label in rules if i % divisor == 0)
        out.append(word or str(i))
    return out


def fizzbuzz_lazy(rules: Sequence[tuple[int, str]] = DEFAULT_RULES) -> Iterator[str]:
    """Unbounded generator version — take as many as you need."""
    i = 1
    while True:
        word = "".join(label for divisor, label in rules if i % divisor == 0)
        yield word or str(i)
        i += 1


# ----------------------------------------------------------------------------------
# The relatives that get asked in the same breath
# ----------------------------------------------------------------------------------

def is_palindrome(text: str) -> bool:
    """Alphanumeric, case-insensitive palindrome test.

    >>> is_palindrome("A man, a plan, a canal: Panama")
    True
    """
    cleaned = [c.lower() for c in text if c.isalnum()]
    return cleaned == cleaned[::-1]


def fibonacci(n: int) -> list[int]:
    """First n Fibonacci numbers, iteratively (no recursion, no stack blowup).

    >>> fibonacci(7)
    [0, 1, 1, 2, 3, 5, 8]
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    out: list[int] = []
    a, b = 0, 1
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out


def is_prime(n: int) -> bool:
    """Trial division up to sqrt(n), skipping even candidates.

    >>> [x for x in range(20) if is_prime(x)]
    [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    factor = 3
    while factor * factor <= n:
        if n % factor == 0:
            return False
        factor += 2
    return True


def two_sum(numbers: Sequence[int], target: int) -> tuple[int, int] | None:
    """Indices of the two values summing to target, in one pass.

    The naive answer is O(n^2) nested loops; this is O(n) with a seen-map.

    >>> two_sum([2, 7, 11, 15], 9)
    (0, 1)
    """
    seen: dict[int, int] = {}
    for index, value in enumerate(numbers):
        complement = target - value
        if complement in seen:
            return seen[complement], index
        seen[value] = index
    return None


def reverse_words(sentence: str) -> str:
    """Reverse word order, collapsing runs of whitespace.

    >>> reverse_words("  the sky   is blue ")
    'blue is sky the'
    """
    return " ".join(reversed(sentence.split()))


def anagrams(first: str, second: str) -> bool:
    """Case-insensitive anagram test ignoring spaces.

    >>> anagrams("Listen", "Silent")
    True
    """
    normalise = lambda s: sorted(c for c in s.lower() if not c.isspace())
    return normalise(first) == normalise(second)


def main() -> None:
    print("=" * 70)
    print("Classic problems")
    print("=" * 70)

    print("\nFizzBuzz, 1..20 (rule-driven):")
    print("  " + " ".join(fizzbuzz(20)))

    print("\nSame output from the naive branch version:", end=" ")
    print(fizzbuzz(20) == fizzbuzz_naive(20))

    print("\nExtended with Bazz for 7, 1..21:")
    extended = fizzbuzz(21, ((3, "Fizz"), (5, "Buzz"), (7, "Bazz")))
    print("  " + " ".join(extended))
    print(f"  note 21 -> {extended[20]} (3 and 7), 15 -> {extended[14]} (3 and 5)")

    print("\nLazy generator, first 15:")
    gen = fizzbuzz_lazy()
    print("  " + " ".join(next(gen) for _ in range(15)))

    print("\nOthers:")
    print(f"  fibonacci(10)                    -> {fibonacci(10)}")
    print(f"  primes below 30                  -> {[x for x in range(30) if is_prime(x)]}")
    print(f"  is_palindrome('Never odd or even') -> {is_palindrome('Never odd or even')}")
    print(f"  two_sum([2,7,11,15], 26)         -> {two_sum([2, 7, 11, 15], 26)}")
    print(f"  reverse_words('the sky is blue') -> '{reverse_words('the sky is blue')}'")
    print(f"  anagrams('Dormitory','Dirty Room') -> {anagrams('Dormitory', 'Dirty Room')}")


if __name__ == "__main__":
    main()
