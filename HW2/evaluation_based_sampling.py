from daphne import daphne
from tests import is_tol, run_prob_test, load_truth

# According to Algorithm 6, primitives should be imported here
from primitives import primitive_funcs
        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    #TODO: Add user function definitions
    ret, sigma = recursive_eval(ast[0], {}, {})

    return ret, sigma


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
            return e, sigma  # Constant

    # Single value expressions
    elif len(e) == 1:
        # Check if it's a variable
        if e[0] in l:
            return l[e[0]], sigma
        else:
            return e[0], sigma  # Constant

    # Sample expression
    elif e[0] == 'sample':
        return None, sigma #TODO

    # Observe expression
    elif e[0] == 'observe':
        return None, sigma #TODO

    # Let expression
    elif e[0] == 'let':
        return None, sigma #TODO

    # If expression
    elif e[0] == 'if':
        return None, sigma #TODO

    # Multi expression
    else:
        c = list()

        for expression in e[1:]:
            temp_c, sigma = recursive_eval(expression, sigma, l)
            c.append(temp_c)

        #TODO: Add user defined functions
        if e[0] in primitive_funcs:
            return primitive_funcs[e[0]](*c), sigma

    return e, sigma


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
    


def run_deterministic_tests():
    
    for i in range(1, 14):

        ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))

        ret, sig = evaluate_program(ast)

        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1, 7):

        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
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
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])