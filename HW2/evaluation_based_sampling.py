from daphne import daphne
from tests import is_tol, run_prob_test, load_truth
import torch

# According to Algorithm 6, primitives should be imported here
from primitives import primitive_funcs

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
            # Change to tensor if int or float
            if type(e) is int or type(e) is float:
                return torch.tensor(e, dtype=torch.float32), sigma  # Constant
            else:
                return e, sigma  # Constant

    # Single value expressions
    elif len(e) == 1:
        # Check if it's a variable
        if e[0] in l:
            return l[e[0]], sigma
        else:
            # Change to tensor if int or float
            if type(e[0]) is int or type(e[0]) is float:
                return torch.tensor(e, dtype=torch.float32), sigma  # Constant
            else:
                return e[0], sigma  # Constant

    # Sample expression
    elif e[0] == 'sample':
        c, sigma = recursive_eval(e[1], sigma, l)
        return c.sample(), sigma

    # Observe expression
    elif e[0] == 'observe':
        # For now let's treat as sampling
        c, sigma = recursive_eval(e[1], sigma, l)
        return c.sample(), sigma

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

    run_deterministic_tests()
    
    run_probabilistic_tests()

    for i in range(1, 5):
        ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast))
