import torch
import torch.distributions as dist
from torch.distributions import Uniform, Beta, Bernoulli, Exponential, Categorical, Gamma, Dirichlet, Cauchy
import torch.functional as F
from copy import deepcopy


def add(alpha, a, b):
    """
    Performs addition

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a+b
    """

    return a + b


def sub(alpha, a, b):
    """
    Performs subtraction

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a-b
    """

    return a - b


def div(alpha, a, b):
    """
    Performs division

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a/b
    """

    return a / b


def mul(alpha, a, b):
    """
    Performs multiplication

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a*b
    """

    return a * b


def sqrt(alpha, a):
    """
    Performs square root

    Args:
        alpha: stack alpha
        a: argument

    Returns:
        sqrt(a)
    """

    return torch.sqrt(a)


def mod(alpha, a, b):
    """
    Performs modulus

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a%b
    """

    return a % b


def pow_op(alpha, a, b):
    """
    Performs exponentiation

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a**b
    """

    return a ** b


def exp(alpha, a):
    """
    Perform e^a

    Args:
        alpha: stack alpha
        a: exponent

    Returns:
        e^a
    """

    return torch.exp(a)


def abs_func(alpha, a):
    """
    return |a|

    Args:
        alpha: stack alpha
        a: Torch tensor to find the absolute value for

    Returns:
        |a|
    """

    return torch.abs(a)


def log(alpha, a):
    """
    return log(a)

    Args:
        alpha: stack alpha
        a: Torch tensor to find log for

    Returns:
        log(a)
    """

    return torch.log(a)


def eq(alpha, a, b):
    """
    return a==b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a==b
    """

    return a == b


def neq(alpha, a, b):
    """
    return a!=b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a!=b
    """

    return a != b


def gt(alpha, a, b):
    """
    Return a>b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a>b
    """

    return a > b


def gteq(alpha, a, b):
    """
    Return a >= b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a>=b
    """

    return a >= b


def sm(alpha, a, b):
    """
    Return a<b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a<b
    """

    return a < b


def smeq(alpha, a, b):
    """
    Return a <= b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a<=b
    """

    return a <= b


def and_op(alpha, a, b):
    """
    Return a and b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a and b
    """

    return a and b


def or_op(alpha, a, b):
    """
    Return a or b

    Args:
        alpha: stack alpha
        a: first argument
        b: second argument

    Returns:
        a or b
    """

    return a or b


def not_op(alpha, a):
    """
    Return ~a

    Args:
        alpha: stack alpha
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
    if len(args[1:]) == 0:
        return torch.tensor(args[1:])
    try:
        return torch.stack(list(args[1:]))
    except:
        return list(args[1:])


def create_list(*args):
    """
    Creates a list

    Args:
        *args: elements of list

    Returns:
        List of arguments
    """

    if len(args[1:]) == 0:
        return list()

    return list(list(args[1:]))


def create_dict(*args):
    """
    Creates dict

    Args:
        *args: Pairs of key values

    Returns:
        dictionary/hashmap
    """

    args = list(args[1:])

    # Change keys from tensor to python numbers
    keys = [key.item() if torch.is_tensor(key) else key for key in args[::2]]

    return dict(zip(keys, args[1::2]))


def first(alpha, a):
    """
    Returns first elements

    Args:
        alpha: stack alpha
        a: list or vector

    Returns:
        First element
    """

    return a[0]


def second(alpha, a):
    """
    Returns the second element

    Args:
        alpha: stack alpha
        a: list or vector

    Returns:
        Second element
    """

    return a[1]


def rest(alpha, a):
    """
    Returns elements except for first

    Args:
        alpha: stack alpha
        a: list or vector

    Returns:
        All elements expect for first
    """

    return a[1:]


def last(alpha, a):
    """
    Returns last elements

    Args:
        alpha: stack alpha
        a: list or vector

    Returns:
        Last element
    """

    return a[-1]


def peek(alpha, a):
    """
    Returns last element of vector and first of list

    Args:
        alpha: stack alpha
        a: list or vector

    Returns:
        Last element of vector or first element of list
    """

    if type(a) is list:
        return a[0]
    else:
        return a[-1]


def nth(alpha, a, n):
    """
    Returns nth element of a

    Args:
        alpha: stack alpha
        a: List or vector
        n: Element index

    Returns:
        nth element
    """

    if torch.is_tensor(n):
        n = int(n.item())

    return a[n]


def conj(alpha, a, b):
    """
    Conj b to a

    Args:
        alpha: stack alpha
        a: list or vector
        b: single value, list or vector

    Returns:
        b appended to a
    """

    # Different behaviour when a is a list or a tensor
    if a is not None:
        if type(a) is list:
            if b:
                if torch.is_tensor(b):
                    b = b.detach().tolist()
                if type(b) is not list:
                    b = [b]
                return b.extend(a)
            else:
                return a
        else:
            if a.dim() == 0:
                a = a.reshape(1)
            if b.dim() == 0:
                b = b.reshape(1)

            return torch.cat((a, b), dim=0)
    else:
        return b


def cons(alpha, a, b):
    """
    cons b to a

    Args:
        alpha: stack alpha
        a:  single value, list or vector
        b: list or vector

    Returns:
        a prepended to b
    """

    if a is not None:
        if type(a) is list:
            if b:
                if torch.is_tensor(b):
                    b = b.detach().tolist()
                if type(b) is not list:
                    b = [b]
                return b.extend(a)
            else:
                return a
        else:
            if a.dim() == 0:
                a = a.reshape(1)
            if b.dim() == 0:
                b = b.reshape(1)

            return torch.cat((b, a), dim=0)
    else:
        return b


def append(alpha, a, b):
    """
    appends b to a

    Args:
        alpha: stack alpha
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


def get(alpha, a, n):
    """
    Get n element from list or dict

    Args:
        alpha: stack alpha
        a: List, vector or dict
        n: index

    Returns:
        a[n]
    """

    if torch.is_tensor(n):
        n = int(n.item())

    return a[n]


def put(alpha, a, n, b):
    """
    Puts b at nth index in a

    Args:
        alpha: stack alpha
        a: List or dict
        n: index
        b: single element

    Returns:
        a with nth element replaced with b
    """

    # Need to make a new dict since we need pure data structs
    new_a = deepcopy(a)

    if torch.is_tensor(n):
        n = int(n.item())

    new_a[n] = b
    return new_a

def empty(alpha, a):
    """
    Checks if torch tensor is empty

    Args:
        alpha: stack alpha
        a: torch tensor

    Returns:
        bool indicating if tensor is empty
    """

    return torch.tensor(a.nelement() == 0)


# def normal(mu, sig):
#     """
#     Return a normal dist with mean mu and std sig
#
#     Args:
#         mu: mean
#         sig: std
#
#     Returns:
#         Return dist object
#     """
#
#     return Normal(mu, sig)

class Normal(dist.Normal):

    def __init__(self, alpha, loc, scale):

        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()

        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """

        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]

        return Normal(*ps)

    def log_prob(self, x):

        self.scale = torch.nn.functional.softplus(self.optim_scale)

        return super().log_prob(x)


def uniform(alpha, start, end):
    """
    Return a uniform over [start, end]

    Args:
        alpha: stack alpha
        start: starting value
        end: last value

    Returns:
        Return dist object
    """

    return Uniform(start, end)


def beta(alpha, a, b):
    """
    Beta distribution

    Args:
        alpha: stack alpha
        a: conc 0
        b: conc 1

    Returns:
        Return dist object
    """

    return Beta(a, b)


def bernoulli(alpha, p):
    """
    Bernoulli distribution

    Args:
        alpha: stack alpha
        p: prob

    Returns:
        Return dist object
    """

    return Bernoulli(p)


def exponential(alpha, r):
    """
    Exponential distribution

    Args:
        alpha: stack alpha
        r: rate

    Returns:
        Return dist object
    """

    return Exponential(r)


def discrete(alpha, p):
    """
    Categorical distribution

    Args:
        alpha: stack alpha
        p: torch tensor of probs

    Returns:
        dist object
    """

    return Categorical(p)


def gamma(alpha, a, b):
    """
    Gamma distribution

    Args:
        alpha: stack alpha
        a: concentration
        b: rate

    Returns:
        Return dist object
    """

    return Gamma(a, b)


def dirichlet(alpha, a):
    """
    Dirichlet distribution

    Args:
        alpha: stack alpha
        a: concentration

    Returns:
        Return dist object
    """

    return Dirichlet(a)


def matmul(alpha, a, b):
    """
    Matrix multiplication

    Args:
        alpha: stack alpha
        a: torch tensor
        b: torch tensor

    Returns:
        a*b
    """

    return torch.matmul(a, b)


class Dirac_approx:
    """
    An implementation for dirac distribution
    """

    def __init__(self, alpha, a):
        """
        Constructor

        Args:
            a: value where prob is 1
        """
        self.center = a

        self.dist = Cauchy(a, 0.1)

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

        return self.dist.log_prob(b)


class Dirac:
    """
    An implementation for dirac distribution
    """

    def __init__(self, alpha, a):
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
            return torch.tensor(0, dtype=torch.float32)
        else:
            return torch.tensor(float('-inf'))


def uniform_cont(*args):
    """
    An approximation for uniform with positive real line support

    Returns:
        Gamma dist
    """
    return Gamma(concentration=torch.tensor(1.0), rate=torch.tensor(2.0))


def matadd(alpha, a, b):
    """
    Matrix Addition

    Args:
        alpha: stack alpha
        a: torch tensor
        b: torch tensor

    Returns:
        a+b
    """

    return a + b


def mattranspose(alpha, a):
    """
    Transpose matrix

    Args:
        alpha: stack alpha
        a: torch tensor

    Returns:
        Tranposed a
    """

    return a.T


def mattanh(alpha, a):
    """
    Tanh activation on matrix

    Args:
        alpha: stack alpha
        a: torch tensor

    Returns:
        tanh(a)
    """

    return torch.tanh(a)


def matrelu(alpha, a):
    """
    ReLu activation on matrix

    Args:
        alpha: stack alpha
        a: torch tensor

    Returns:
        relu(a)
    """

    return F.relu(a)


def matrepmat(alpha, a, n, m):
    """
    Repeats a n*m times

    Args:
        alpha: stack alpha
        a: torch tensor
        n: torch tensor
        m: torch tensor

    Returns:
        a repated over n rows and m columns
    """

    return a.repeat(int(n.item()), int(m.item()))
        

def push_addr(alpha, value):
    return alpha + value


env = {'push-address': push_addr,
       '+': add,  # Math operations
       '-': sub,
       '/': div,
       '*': mul,
       'sqrt': sqrt,
       '%': mod,
       '**': pow_op,
       'exp': exp,
       'abs': abs_func,
       'log': log,
       '=': eq, # Logical Operations
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
       'peek': peek,
       'nth': nth,
       'conj': conj,
       'cons': cons,
       'append': append,
       'get': get,
       'put': put,
       'empty?': empty,
       'mat-mul': matmul,  # Matrix operations
       'mat-add': matadd,
       'mat-transpose': mattranspose,
       'mat-tanh': mattanh,
       'mat-relu': matrelu,
       'mat-repmat': matrepmat,
       'normal': Normal,  # Distributions
       'uniform': uniform,
       'uniform-continuous': uniform,
       'beta': beta,
       'bernoulli': bernoulli,
       'exponential': exponential,
       'discrete': discrete,
       'gamma': gamma,
       'dirichlet': dirichlet,
       'flip': bernoulli,
       'dirac': Dirac_approx}
