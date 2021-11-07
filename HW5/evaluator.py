
import sys
from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test, load_truth
from pyrsistent import pmap
import torch


class Env:
    """
        Environment class that allows for inner environments to be searched before outermost ones
        This is copied from https://norvig.com/lispy.html and modified to use pure data structures
    """

    def __init__(self, init_dict, outer=None):
        self.env_dict = pmap(init_dict)
        self.outer = outer

        # We need to add alpha here since procedure would make calls to this as well
        self.env_dict = self.env_dict.update({'alpha': ''})

    def find(self, var):
        return self.env_dict[var] if (var in self.env_dict) else self.outer.find(var)

    def has(self, var):
        # Check if variable is in inner environment
        if var in self.env_dict:
            return True

        # Check if variable is in outer environment
        try:
            if self.outer.has(var):
                return True
            else:
                return False
        except:
            return False


class Procedure(object):
    """
        The procedure class copied from https://norvig.com/lispy.html
        Modifed a bit to fit our evaluator
    """
    def __init__(self, params, body, env):
        self.params, self.body, self.env = params, body, env

    def __call__(self, *args, sigma):
        return recursive_eval(self.body, sigma, Env(dict(zip(self.params, args)), self.env))


def standard_env():

    env = Env(penv)
    return env


def recursive_eval(exp, sigma, env=None):

    if env is None:
        env = standard_env()

    # Constants
    if not isinstance(exp, list):
        # Check if it's a variable
        if env.has(exp):
            return env.find(exp), sigma
        else:
            if type(exp) == bool:
                return torch.tensor(exp), sigma
            # Change to tensor if int or float
            elif type(exp) is int or type(exp) is float:
                return torch.tensor(exp, dtype=torch.float32), sigma  # Constant
            else:
                return exp, sigma  # Constant

    # Single value expressions
    elif len(exp) == 1:
        # Check if it's a variable
        if env.has(env[0]):
            return env.has(exp[0]), sigma
        else:
            if type(exp) == bool:
                return torch.tensor(exp), sigma
            # Change to tensor if int or float
            if type(exp[0]) is int or type(exp[0]) is float:
                return torch.tensor(exp, dtype=torch.float32), sigma  # Constant
            else:
                return exp[0], sigma  # Constant

    # Sample expression
    elif exp[0] == 'sample':

        # Evaluate the address expression (this is not used for now)
        _, sigma = recursive_eval(exp[1], sigma, env)

        # Evaluate the expression to a dist object
        d, sigma = recursive_eval(exp[2], sigma, env)

        # Sample from the distribution
        return d.sample(), sigma

    # Observe expression
    elif exp[0] == 'observe':

        # Evaluate the address expression (the address is not really used for now)
        _, sigma = recursive_eval(exp[1], sigma, env)

        # Evaluate expression to obtain the dist object
        d, sigma = recursive_eval(exp[2], sigma, env)

        # Evaluate expression a constant, which is the observed value of y
        c, sigma = recursive_eval(exp[3], sigma, env)

        # This is skipped for now
        # Aggregate the log probability of c under d into logW
        # sigma['logW'] += d.log_prob(c * 1.0)

        return c, sigma

    # If expression
    elif exp[0] == 'if':
        c, sigma = recursive_eval(exp[1], sigma, env)

        if c:
            return recursive_eval(exp[2], sigma, env)
        else:
            return recursive_eval(exp[3], sigma, env)

    # Lambda expressions
    elif exp[0] == 'fn':
        return Procedure(params=exp[1], body=exp[2], env=env), sigma

    # Multi expression
    else:

        # Get the function
        func, sigma = recursive_eval(exp[0], sigma, env)

        c = list()

        # The first parameter in all functions is now an address argument (which can be treated like other params)
        for expression in exp[1:]:
            temp_c, sigma = recursive_eval(expression, sigma, env)
            c.append(temp_c)

        # Check if we have a procedure
        if type(func) == Procedure:
            return func(*c, sigma=sigma)
        else:
            return func(*c), sigma


def evaluate(exp):

        # This returns a procedure and not the evaluated value
        func, sigma = recursive_eval(exp, sigma=pmap({}))
        ret, sigma = func(sigma=sigma)
        return ret


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    
    for i in range(1, 14):

        exp = daphne(['desugar-hoppl', '-i', '../cpsc532w_hw/HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

        print('FOPPL Tests passed')
        
    for i in range(1, 13):

        exp = daphne(['desugar-hoppl', '-i', '../cpsc532w_hw/HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples = 1e4
    max_p_value = 1e-2
    
    for i in range(1, 7):
        exp = daphne(['desugar-hoppl', '-i', '../cpsc532w_hw/HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1, 4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../cpsc532w_hw/HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate(exp))
