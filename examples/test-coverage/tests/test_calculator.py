"""Starter tests — deliberately incomplete coverage."""

from src.calculator import add, divide, factorial


def test_add_positive():
    assert add(2, 3) == 5


def test_divide_normal():
    assert divide(10, 2) == 5.0


def test_factorial_five():
    assert factorial(5) == 120
