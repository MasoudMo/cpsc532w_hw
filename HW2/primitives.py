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

    return a + b


def sub(a, b):
    """
    Performs subtraction

    Args:
        a: first argument
        b: second argument

    Returns:
        a-b
    """

    return a - b


def div(a, b):
    """
    Performs division

    Args:
        a: first argument
        b: second argument

    Returns:
        a/b
    """

    return a / b


def mul(a, b):
    """
    Performs multiplication

    Args:
        a: first argument
        b: second argument

    Returns:
        a*b
    """

    return a * b


def sqrt(a):
    """
    Performs square root

    Args:
        a: argument

    Returns:
        sqrt(a)
    """

    return torch.sqrt(a)


def mod(a, b):
    """
    Performs modulus

    Args:
        a: first argument
        b: second argument

    Returns:
        a%b
    """

    return a % b


def pow_op(a, b):
    """
    Performs exponentiation

    Args:
        a: first argument
        b: second argument

    Returns:
        a**b
    """

    return a ** b


def eq(a, b):
    """
    return a==b

    Args:
        a: first argument
        b: second argument

    Returns:
        a==b
    """

    return a == b


def neq(a, b):
    """
    return a!=b

    Args:
        a: first argument
        b: second argument

    Returns:
        a!=b
    """

    return a != b


def gt(a, b):
    """
    Return a>b

    Args:
        a: first argument
        b: second argument

    Returns:
        a>b
    """

    return a > b


def gteq(a, b):
    """
    Return a >= b

    Args:
        a: first argument
        b: second argument

    Returns:
        a>=b
    """

    return a >= b


def sm(a, b):
    """
    Return a<b

    Args:
        a: first argument
        b: second argument

    Returns:
        a<b
    """

    return a < b


def smeq(a, b):
    """
    Return a <= b

    Args:
        a: first argument
        b: second argument

    Returns:
        a<=b
    """

    return a <= b


def and_op(a, b):
    """
    Return a and b

    Args:
        a: first argument
        b: second argument

    Returns:
        a and b
    """

    return a and b


def or_op(a, b):
    """
    Return a or b

    Args:
        a: first argument
        b: second argument

    Returns:
        a or b
    """

    return a or b


def not_op(a):
    """
    Return ~a

    Args:
        a: argument

    Returns:
        not a
    """

    return not a


def create_vec(*args):
    """
    Creates a vector

    Args:
        *args: elements of vector

    Returns:
        Tensor array of arguments
    """

    return torch.tensor(list(args))


def create_list(*args):
    """
    Creates a list

    Args:
        *args: elements of list

    Returns:
        List of arguments
    """

    return list(args)


def create_dict(*args):
    """
    Creates dict

    Args:
        *args: Pairs of key values

    Returns:
        dictionary/hashmap
    """

    args = list(args)

    # Change keys from tensor to python numbers
    keys = [key.item() if torch.is_tensor(key) else key for key in args[::2]]

    return dict(zip(keys, args[1::2]))


def first(a):
    """
    Returns first elements

    Args:
        a: list or vector

    Returns:
        First element
    """

    return a[0]


def rest(a):
    """
    Returns elements except for first

    Args:
        a: list or vector

    Returns:
        All elements expect for first
    """

    return a[1:]


def last(a):
    """
    Returns last elements

    Args:
        a: list or vector

    Returns:
        Last element
    """

    return a[-1]


def nth(a, n):
    """
    Returns nth element of a

    Args:
        a: List or vector
        n: Element index

    Returns:
        nth element
    """

    return a[n]


def conj(a, b):
    """
    Conj b to a

    Args:
        a: list or vector
        b: single value, list or vector

    Returns:
        a and b prepended or appended
    """

    if type(a) is list:
        if type(b) is list:
            return b.extend(a)
        else:
            return list(b).extend(a)

    else:
        return torch.cat((a, b))


def cons(a, b):
    """
    cons b to a

    Args:
        a: list or vector
        b: single value, list or vector

    Returns:
        b prepended to a
    """

    if type(a) is list:
        if type(b) is list:
            return b.extend(a)
        else:
            return list(b).extend(a)

    else:
        return torch.cat((b, a))


def append(a, b):
    """
    appends b to a

    Args:
        a: list or vector
        b: single value, list or vector

    Returns:
        b appended to a
    """

    if type(a) is list:
        if type(b) is list:
            return a.extend(b)
        else:
            return a.extend(list(b))

    else:
        if torch.is_tensor(b):
            return torch.cat((a, b.view(1)))
        else:
            return torch.cat((a, torch.tensor([b])), dim=0)


def get(a, n):
    """
    Get n element from list or dict

    Args:
        a: List, vector or dict
        n: index

    Returns:
        a[n]
    """

    if torch.is_tensor(n):
        n = n.item()

    return a[n]


def put(a, n, b):
    """
    Puts b at nth index in a

    Args:
        a: List
        n: index
        b: single element

    Returns:
        a with nth element replaced with b
    """

    if torch.is_tensor(n):
        n = n.item()

    a[n] = b
    return a


# Mapping between strings and primitive functions
primitive_funcs = {'+': add,  # Math operations
                   '-': sub,
                   '/': div,
                   '*': mul,
                   'sqrt': sqrt,
                   '%': mod,
                   '**': pow_op,
                   '==': eq,  # Logical Operations
                   '!=': neq,
                   '>': gt,
                   '>=': gteq,
                   '<': sm,
                   '<=': smeq,
                   'and': and_op,
                   'or': or_op,
                   'not': not_op,
                   'vector': create_vec,  # Data structures
                   'list': create_list,
                   'hash-map': create_dict,
                   'first': first,  # Data structure methods
                   'rest': rest,
                   'last': last,
                   'nth': nth,
                   'conj': conj,
                   'cons': cons,
                   'append': append,
                   'get': get,
                   'put': put} #TODO: Add mat operations

# TODO: Add distributions
