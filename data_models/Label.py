from enum import IntEnum


class Label(IntEnum):
    """
    Enum class for mapping nmsc types to integer labels.
    """

    na = 0
    bowens = 1
    bcc = 2
    scc = 3
