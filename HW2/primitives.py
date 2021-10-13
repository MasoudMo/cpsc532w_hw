import torch


def add(a, b):
    """
    Performs addition

    Args:
        a: first argument
        b: second argument

    Returns:
        a+b
    """

    return torch.tensor(a+b)

# Mapping between strings and primitive functions
primitive_funcs = {'+': add}
