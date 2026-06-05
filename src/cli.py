import argparse

def positive_int(value: str) -> int:
    # ---------------------------------------------------------
    # Accept only positive integer values from CLI.
    # ---------------------------------------------------------
    count = int(value)

    if count <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")

    return count


def non_negative_int(value: str) -> int:
    # ---------------------------------------------------------
    # Accept only zero or positive integer values from CLI.
    # ---------------------------------------------------------
    count = int(value)

    if count < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")

    return count
