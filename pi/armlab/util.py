"""
Small helpers.
"""


def ask(prompt):
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return "q"
