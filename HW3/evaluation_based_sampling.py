from daphne import daphne
from tests import is_tol, run_prob_test, load_truth
import torch
from primitives import primitive_funcs
import numpy as np
import time

# Dict containing user-defined functions
user_funcs = dict()


def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """

    # Go over expressions and see if any user defined functions are present
    # Note: Only definitions before the main body are considered
    idx = 0
    while ast[idx][0] == 'defn':
        user_funcs.update({ast[idx][1]: {'args': ast[idx][2], 'body': ast[idx][3]}})
        idx += 1

    ret, sigma = recursive_eval(ast[idx], {}, {})

    return ret


def evaluate_likelihood_weighting(ast, num_iters):
    """
    Perform weighted sampling with the prior p(x) as the proposed distribution q(x)

    Args:
        ast: Expression to evaluate
        num_iters: Number of iteration (samples to generated)

    Returns:
        Tuples containing samples and their weights
    """

    # Go over expressions and see if any user defined functions are present
    # Note: Only definitions before the main body are considered
    idx = 0
    while ast[idx][0] == 'defn':
        user_funcs.update({ast[idx][1]: {'args': ast[idx][2], 'body': ast[idx][3]}})
        idx += 1

    # List containing output sample and weight tuples
    samples = list()

    for l in range(num_iters):
        ret, sigma = recursive_eval(ast[idx], {'logW': 0}, {})
        samples.append((ret, sigma['logW']))

    return samples


def compute_identity_is_expectation(weighted_samples):
    """
    Computes posterior expection for identity function

    Args:
        weighted_samples: list of tuples of sample and weight pairs

    Returns:
        Expectation of posterior for identity function
    """

    # Extract the samples
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in weighted_samples])

    # Ensure correct shape for multiplication with W
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1)

    # Obtain log weights and exp them to obtain actual weights
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    weights = np.expand_dims(weights, axis=1)

    # Perform weighted average
    return np.sum((samples * weights) / np.sum(weights), axis=0)


def compute_identity_is_variance(weighted_samples, mean):
    """
    Computes posterior expection for identity function

    Args:
        weighted_samples: list of tuples of sample and weight pairs
        mean: mean of samples

    Returns:
        Expectation of posterior for identity function
    """

    # Extract the samples
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in weighted_samples])

    # Ensure correct shape for multiplication with W
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1)

    # Obtain log weights and exp them to obtain actual weights
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    weights = np.expand_dims(weights, axis=1)

    # Perform weighted average
    return np.sum(((samples ** 2 - mean ** 2) * weights) / np.sum(weights), axis=0)


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

        # This trick would ensure that true/false values are changed into 0 and 1
        c = c * 1.0

        # Aggregate the log probability of c under d into logW
        sigma['logW'] += d.log_prob(c)

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


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)


def run_deterministic_tests():
    
    for i in range(1, 22):

        ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))

        ret = evaluate_program(ast)

        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')


def run_probabilistic_tests():
    
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1, 8):

        ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    # run_deterministic_tests()
    #
    # run_probabilistic_tests()

    for i in range(2, 5):
        ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW3/programs/{}.daphne'.format(i)])

        # Compute program runtime
        t_start = time.time()

        iterations = 10000

        weighted_samples = evaluate_likelihood_weighting(ast, iterations)

        print('It took {} seconds for sampler with {} iterations.'.format((time.time() - t_start), iterations))

        posterior_expectation = compute_identity_is_expectation(weighted_samples)
        variance = compute_identity_is_variance(weighted_samples, posterior_expectation)
        print('The posterior expectation is {}. \n'.format(posterior_expectation))
