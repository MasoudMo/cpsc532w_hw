from daphne import daphne
import torch
from primitives import primitive_funcs
import numpy as np
import time

# Dict containing user-defined functions
user_funcs = dict()


def recursive_eval(e, sigma, l):
    """
    Evaluates a program recursively

    Args:
        e: expression to be evaluated
        sigma: mapping of inference state variables (Needed for storing side effects)
        l: mapping of local variables (Local variables bound in let forms and function calls)

    Returns:
        The evaluation output and state variable mapping
    """

    # Constants
    if not isinstance(e, list):
        # Check if it's a variable
        if e in l:
            return l[e], sigma
        else:
            if e in primitive_funcs:
                return primitive_funcs[e], sigma
            elif type(e) == bool:
                return torch.tensor(e), sigma
            # Change to tensor if int or float
            elif type(e) is int or type(e) is float:
                return torch.tensor(e, dtype=torch.float32), sigma  # Constant
            else:
                return e, sigma  # Constant

    # Single value expressions
    elif len(e) == 1:
        # Check if it's a variable
        if e[0] in l:
            return l[e[0]], sigma
        else:
            if e in primitive_funcs:
                return primitive_funcs[e], sigma
            elif type(e) == bool:
                return torch.tensor(e), sigma
            # Change to tensor if int or float
            if type(e[0]) is int or type(e[0]) is float:
                return torch.tensor(e, dtype=torch.float32), sigma  # Constant
            else:
                return e[0], sigma  # Constant

    # Sample expression
    elif e[0] == 'sample':

        # Evaluate the expression to a dist object
        d, sigma = recursive_eval(e[1], sigma, l)

        # Sample from the distribution
        return d.sample(), sigma

    # Observe expression
    elif e[0] == 'observe':

        # Evaluate e1 to obtain the dist object
        d, sigma = recursive_eval(e[1], sigma, l)

        # Evaluate e2 to a constant, which is the observed value of y
        c, sigma = recursive_eval(e[2], sigma, l)

        # Aggregate the log probability of c under d into logW
        sigma['logW'] += d.log_prob(c * 1.0)

        return c, sigma

    # Let expression
    elif e[0] == 'let':
        # Bind the variable
        c, sigma = recursive_eval(e[1][1], sigma, l)
        l.update({e[1][0]: c})
        return recursive_eval(e[2], sigma, l)

    # If expression
    elif e[0] == 'if':
        c, sigma = recursive_eval(e[1], sigma, l)

        if c:
            return recursive_eval(e[2], sigma, l)
        else:
            return recursive_eval(e[3], sigma, l)

    # Multi expression
    else:
        c = list()

        for expression in e[1:]:
            temp_c, sigma = recursive_eval(expression, sigma, l)
            c.append(temp_c)

        # Check user funcs first as user can overwrite primitives
        if e[0] in user_funcs:
            func = user_funcs[e[0]]
            args_dict = dict(zip(func['args'], c))
            # Merging two dicts: https://www.geeksforgeeks.org/python-merging-two-dictionaries/
            return recursive_eval(func['body'], sigma, {**l, **args_dict})

        # Primitive funcs
        elif e[0] in primitive_funcs:
            return primitive_funcs[e[0]](*c), sigma

        else:
            return None, sigma

