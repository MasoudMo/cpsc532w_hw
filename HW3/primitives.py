import torch
from torch.distributions import Normal, Uniform, Beta, Bernoulli, Exponential, Categorical, Gamma, Dirichlet
import torch.nn.functional as F


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


def exp(a):
    """
    Perform e^a

    Args:
        a: exponent

    Returns:
        e^a
    """

    return torch.exp(a)


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

    # May have dist objects as elements therefore tensor cannot always be used.
    try:
        return torch.stack(list(args))
    except:
        return list(args)


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


def second(a):
    """
    Returns the second element

    Args:
        a: list or vector

    Returns:
        Second element
    """

    return a[1]


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

    if torch.is_tensor(n):
        n = int(n.item())
        
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
        n = int(n.item())

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
        n = int(n.item())

    a[n] = b
    return a


def normal(mu, sig):
    """
    Return a normal dist with mean mu and std sig

    Args:
        mu: mean
        sig: std

    Returns:
        Return dist object
    """

    return Normal(mu, sig)


def uniform(start, end):
    """
    Return a uniform over [start, end]

    Args:
        start: starting value
        end: last value

    Returns:
        Return dist object
    """

    return Uniform(start, end)


def beta(a, b):
    """
    Beta distribution

    Args:
        a: conc 0
        b: conc 1

    Returns:
        Return dist object
    """

    return Beta(a, b)


def bernoulli(p):
    """
    Bernoulli distribution

    Args:
        p: prob

    Returns:
        Return dist object
    """

    return Bernoulli(p)


def exponential(r):
    """
    Exponential distribution

    Args:
        r: rate

    Returns:
        Return dist object
    """

    return Exponential(r)


def discrete(p):
    """
    Categorical distribution

    Args:
        p: torch tensor of probs

    Returns:
        dist object
    """

    return Categorical(p)


def gamma(a, b):
    """
    Gamma distribution

    Args:
        a: concentration
        b: rate

    Returns:
        Return dist object
    """

    return Gamma(a, b)


def dirichlet(a):
    """
    Dirichlet distribution

    Args:
        a: concentration

    Returns:
        Return dist object
    """

    return Dirichlet(a)


def matmul(a, b):
    """
    Matrix multiplication

    Args:
        a: torch tensor
        b: torch tensor

    Returns:
        a*b
    """

    return torch.matmul(a, b)


class Dirac:
    """
    An implementation for dirac distribution
    """

    def __init__(self, a):
        """
        Constructor

        Args:
            a: value where prob is 1
        """
        self.center = a

    def sample(self):
        """
        sample method

        Returns:
            Simply returns the center value as all other values have a probability of 0
        """
        return self.center

    def log_prob(self, b):
        """
        Log of likelihood probability

        Args:
            b: Value of observed

        Returns:
            1 if b is equal to center, 0 otherwise
        """

        if self.center == b:
            return torch.log(torch.tensor(1))
        else:
            return torch.tensor(-float('Inf'))


def matadd(a, b):
    """
    Matrix Addition

    Args:
        a: torch tensor
        b: torch tensor

    Returns:
        a+b
    """

    return a + b


def mattranspose(a):
    """
    Transpose matrix

    Args:
        a: torch tensor

    Returns:
        Tranposed a
    """

    return a.T


def mattanh(a):
    """
    Tanh activation on matrix

    Args:
        a: torch tensor

    Returns:
        tanh(a)
    """

    return torch.tanh(a)


def matrelu(a):
    """
    ReLu activation on matrix

    Args:
        a: torch tensor

    Returns:
        relu(a)
    """

    return F.relu(a)


def matrepmat(a, n, m):
    """
    Repeats a n*m times

    Args:
        a: torch tensor
        n: torch tensor
        m: torch tensor

    Returns:
        a repated over n rows and m columns
    """

    return a.repeat(int(n.item()), int(m.item()))


# Mapping between strings and primitive functions
primitive_funcs = {'+': add,  # Math operations
                   '-': sub,
                   '/': div,
                   '*': mul,
                   'sqrt': sqrt,
                   '%': mod,
                   '**': pow_op,
                   'exp': exp,
                   '=': eq,  # Logical Operations
                   '!=': neq,
                   '>': gt,
                   '>=': gteq,
                   '<': sm,
                   '<=': smeq,
                   'and': and_op,
                   'or': or_op,
                   'not': not_op,
                   'true': torch.tensor(True),
                   'false': torch.tensor(False),
                   'vector': create_vec,  # Data structures
                   'list': create_list,
                   'hash-map': create_dict,
                   'first': first,  # Data structure methods
                   'second': second,
                   'rest': rest,
                   'last': last,
                   'nth': nth,
                   'conj': conj,
                   'cons': cons,
                   'append': append,
                   'get': get,
                   'put': put,
                   'mat-mul': matmul,  # Matrix operations
                   'mat-add': matadd,
                   'mat-transpose': mattranspose,
                   'mat-tanh': mattanh,
                   'mat-relu': matrelu,
                   'mat-repmat': matrepmat,
                   'normal': normal,  # Distributions
                   'uniform': uniform,
                   'beta': beta,
                   'bernoulli': bernoulli,
                   'exponential': exponential,
                   'discrete': discrete,
                   'gamma': gamma,
                   'dirichlet': dirichlet,
                   'flip': bernoulli,
                   'dirac': Dirac}
