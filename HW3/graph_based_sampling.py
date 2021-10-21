import torch
import torch.distributions as dist

from daphne import daphne

from primitives import primitive_funcs
from tests import is_tol, run_prob_test, load_truth
from evaluation_based_sampling import recursive_eval, user_funcs

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = primitive_funcs


# Evaluator for link functions
def evaluate_link_func(expression, local_env, user_defns):
    """
    Evaluates the link functions

    Args:
        user_defns: Dictionary of user definitions
        expression: Expression to be evaluated
        local_env: Dict containing variable bindings

    Returns:
        Expression return value
    """

    for func in user_defns.items():
        user_funcs.update({func[0]: {'args': func[1][1], 'body': func[1][2]}})

    ret, sigma = recursive_eval(expression, {}, local_env)

    return ret


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*map(deterministic_eval, args))
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        raise("Expression type unknown.", exp)


# We should implement topological sort to perform ancestral sampling
# I implement the DFS algorithm outlined here: https://en.wikipedia.org/wiki/Topological_sorting
def top_sort(nodes, edges):
    """
    Topologically sorts graph nodes

    Args:
        nodes: Nodes of the graph as outputted by daphne
        edges: Dict showing edges from each node as returned by daphne

    Returns:
        topologically sorted nodes
    """

    # Inner recursive function for DFS
    def dfs_visit(node):
        """
        Perform DFS for topological sort

        Args:
            node: Node to start traversing from
        """
        if perm_visited[node]:
            return

        assert not temp_visited[node], 'The graph is not DAG'

        temp_visited[node] = True

        if node in edges.keys():
            for child in edges[node]:
                dfs_visit(child)

        temp_visited[node] = False
        perm_visited[node] = True
        sorted_nodes.insert(0, node)

        return

    sorted_nodes = list()
    temp_visited = dict.fromkeys(nodes, False)
    perm_visited = dict.fromkeys(nodes, False)

    while False in perm_visited.values():
        for key in perm_visited:
            if not perm_visited[key]:
                dfs_visit(key)

    return sorted_nodes


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."


    # Extract graph dics
    user_defns = graph[0]
    nodes = graph[1]['V']
    edges = graph[1]['A']
    link_funcs = graph[1]['P']
    return_expression = graph[2]

    # To perform ancestral sampling, we should start from nodes with no dependence
    # This is achieved by performing topological sampling
    sorted_nodes = top_sort(nodes, edges)

    # Dictionary holding values for sampled nodes
    node_values = dict.fromkeys(nodes, None)

    for node in sorted_nodes:

        if link_funcs[node][0] == 'sample*':
            node_values[node] = evaluate_link_func(link_funcs[node][1], node_values, user_defns).sample()
        elif link_funcs[node][0] == 'observe*':
            # For now, we treat observe in a similar fashion to sample
            node_values[node] = evaluate_link_func(link_funcs[node][1], node_values, user_defns).sample()
        else:
            print('No support beyond sample and observe for random variable. Return None.')
            return None

    # Evaluate the final return expression using the sampled values of random variables
    ret_val, _ = recursive_eval(return_expression, {}, node_values)

    return ret_val


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)

#Testing:

def run_deterministic_tests():
    
    for i in range(1, 13):

        graph = daphne(['graph', '-i', '../cpsc532w_hw/HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    

def run_probabilistic_tests():
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1, 7):

        graph = daphne(['graph', '-i', '../cpsc532w_hw/HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')
        
        
if __name__ == '__main__':

    run_deterministic_tests()
    run_probabilistic_tests()

    for i in range(1, 5):
        graph = daphne(['graph', '-i', '../cpsc532w_hw/HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    
