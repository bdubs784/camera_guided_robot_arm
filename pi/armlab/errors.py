"""The two kinds of failure every part of the program raises."""


class ArmError(Exception):
    """Something stopped the run: the message says what, and what to do about it."""


