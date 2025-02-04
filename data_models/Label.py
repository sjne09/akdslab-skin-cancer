from enum import IntEnum


class Label(IntEnum):
    """
    Enum class for mapping nmsc types to integer labels.
    """

    na = 0
    bowens = 1
    bcc = 2
    scc = 3


class NCLabel(IntEnum):
    """
    Enum class for mapping nmsc labels for nearest centroids to integer
    labels.
    """

    dermis_subcutis = 0
    epidermis_corneum = 1
    bowens = 2
    bcc_nodular = 3
    bcc_superficial = 4
    scc = 5
    artifact = 6
