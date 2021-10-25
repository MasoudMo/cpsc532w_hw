import torch
import torch.distributions as dist
from daphne import daphne
import numpy as np
from primitives import primitive_funcs
from tests import is_tol, run_prob_test, load_truth
from evaluation_based_sampling import recursive_eval, user_funcs
import time
import copy

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = primitive_funcs


######################################## MH WITHING GIBBS ##################################################

def flatten_list(alist):
    """
    Return the flattened version of multi-dim list
    Implementation was borrowed from https://stackabuse.com/python-how-to-flatten-list-of-lists/

    Args:
        alist: multi-dim list

    Returns:
        Flattened version of list
    """

    if len(alist) == 0:
        return alist
    if isinstance(alist[0], list):
        return flatten_list(alist[0]) + flatten_list(alist[1:])

    return alist[:1] + flatten_list(alist[1:])


def markov_blanket(graph):
    """
    Find the markov blanket of latent variables in the graph

    Args:
        graph: graph structure as obtained from Daphne

    Returns:
        Markov blanket for the node
    """

    nodes = graph[1]['V']
    edges = graph[1]['A']
    # link_funcs = graph[1]['P']
    observed_nodes = graph[1]['Y']

    # List containing blanket nodes
    mk_blankets = {}

    # Find the blanket for x
    for x in nodes:
        if x not in observed_nodes:
            temp = {x: [x]}

            # parents
            # for term in flatten_list(link_funcs[x]):
            #     if term in nodes:
            #         temp[x].append(term)

            # Find children of x
            for child in edges[x]:
                temp[x].append(child)

                # Find parents of children
                # for term in flatten_list(link_funcs[child]):
                #     if term in nodes and term != x:
                #         temp[x].append(term)

            temp[x] = list(set(temp[x]))
            mk_blankets.update(temp)

    return mk_blankets


def gibbs_accept(graph, node_values, new_node_values, x, V_x):
    """
    Compute the acceptance ratio

    Args:
        graph: Graph structure as obtained from Daphne
        node_values: Old node values dict
        new_node_values: New node values dict
        x: The node to evaluate acceptance ratio for
        V_x: markov blanket for x

    Returns:
        Acceptance ratio
    """

    user_defns = graph[0]
    link_funcs = graph[1]['P']

    # Evaluate a dist object for x --> x'
    d = evaluate_link_func(link_funcs[x][1], node_values, user_defns)

    # Evaluate a dist object for x' --> x
    d_p = evaluate_link_func(link_funcs[x][1], new_node_values, user_defns)

    log_alpha = d_p.log_prob(node_values[x]) - d.log_prob(new_node_values[x])

    # Sample from joint of blanket
    for v in V_x:
        log_alpha += evaluate_link_func(link_funcs[v][1], new_node_values,
                                        user_defns).log_prob(new_node_values[v] * 1.0)
        log_alpha -= evaluate_link_func(link_funcs[v][1], node_values,
                                        user_defns).log_prob(node_values[v] * 1.0)

    return torch.exp(log_alpha)


def gibbs_step(graph, node_values, mk_blankets):
    """
    Performs one step of Gibbs sampling

    Args:
        graph: Graph structure as obtained from Daphne
        node_values: Node values dictionary
        mk_blankets: Dict containing markov blankets for each node

    Returns:
        New node values in the graph after Gibbs step
    """

    user_defns = graph[0]
    nodes = graph[1]['V']
    link_funcs = graph[1]['P']
    observed_values = graph[1]['Y']

    uniform_dist = dist.Uniform(0, 1)

    # only traverse through latent variables (not the observed ones)
    for x in nodes:
        if x not in observed_values:
            # Get the proposed distribution from prior
            d = evaluate_link_func(link_funcs[x][1], node_values, user_defns)

            # Copy current node values and replace the newly sampled one
            new_node_values = copy.deepcopy(node_values)
            new_node_values[x] = d.sample()

            # Determine the acceptance ratio
            acceptance_ratio = gibbs_accept(graph, node_values, new_node_values, x, mk_blankets[x])

            # Sample from uniform
            u = uniform_dist.sample()

            # Reject or accept sample
            if u < acceptance_ratio:
                node_values = new_node_values

    return node_values


def gibbs(graph, num_iterations):
    """
    Performs Gibbs sampling

    Args:
        graph: Graph structure as obtained form Daphne
        num_iterations: Number of Gibbs sampling iterations

    Returns:
        Samples
    """

    # Compute the markov blanket for all latent variables (this is done here to save computation time)
    # Alternatively, this markov blanket could be computed each time in gibbs_accept
    mk_blankets = markov_blanket(graph)

    # Need to obtain initial values for latent variables (X[0])
    node_values = sample_from_joint(graph, only_latent_vars=True)

    # list holding sample values
    sampled_node_values = list()
    sampled_node_values.append(node_values)

    # Obtain node values for multiple iterations of Gibbs
    for s in range(num_iterations):
        sampled_node_values.append(gibbs_step(graph, sampled_node_values[-1], mk_blankets))

    return sampled_node_values[1:]

###########################################################################################################

################################################## HMC ####################################################


def hmc_potential(graph, Y, X):
    """
    Compute the potential as -log joint probability of X

    Args:
        graph: graph structure as obtained from Daphne
        Y: Y node values dict
        X: X node values dict

    Returns:
        Potential as log joint
    """

    user_defns = graph[0]
    link_funcs = graph[1]['P']

    # gamma = p(X, Y=observed)
    log_gamma = 0

    node_values = {**X, **Y}
    for node in node_values:
        log_gamma -= evaluate_link_func(link_funcs[node][1], node_values, user_defns).log_prob(node_values[node] * 1.0)

    return log_gamma


def hmc_grad_potential(graph, Y, X):
    """
    Compute the gradient of U with respect to X

    Args:
        graph: graph structure as obtained from Daphne
        Y: Y node values dict
        X: X node values dict

    Returns:
        Gradients with respect to latent variables
    """

    # Forward path
    U = hmc_potential(graph, Y, X)

    # Zero the gradients
    for node in X:
        if X[node].grad is not None:
            X[node].grad.zero_()

    # Run backward
    U.backward()

    # Extract the gradients
    grads = torch.empty(len(X))

    for i, node in enumerate(X):
            grads[i] = X[node].grad

    return grads


def hmc_leap_frog(graph, Y, X, R, T, eps):
    """
    Performs leap frog integration for HMC

    Args:
        graph: graph structure as obtained from Daphne
        Y: Y node values dict
        X: X node values dict
        R: The momentum
        T: Number of steps to take
        eps: The length of the step to take

    Returns:
        New X and momentum values
    """

    # Initial momentum
    R_half_t = R - 0.5 * eps * hmc_grad_potential(graph, Y, X)

    for t in range(T):

        with torch.no_grad():
            for i, node in enumerate(X):
                X[node] = X[node] + eps * R_half_t[i]

        for node in X:
            X[node].requires_grad = True

        R_half_t = R_half_t - eps * hmc_grad_potential(graph, Y, X)

    with torch.no_grad():
        for i, node in enumerate(X):
            X[node] = X[node] + eps * R_half_t[i]

    for node in X:
        X[node].requires_grad = True

    R_half_t = R_half_t - 0.5 * eps * hmc_grad_potential(graph, Y, X)

    return X, R_half_t


def hmc_hamiltonian(graph, Y, X, R, M):
    """
    Computes the Hamiltonian as U+0.5*K

    Args:
        graph: graph structure as obtained from Daphne
        Y: Y node values dict
        X: X node values dict
        R: Momentum
        M: Covariance mass matrix

    Returns:
        The Hamiltonian
    """

    U = hmc_potential(graph, Y, X)


    # In this case M = M_inverse
    K = torch.matmul(R.T, torch.matmul(M, R)) * 0.5

    return U + K


def hmc_accept(graph, Y, X, R, new_X, R_p, M):
    """
    Compute HMC acceptance ratio

    Args:
        graph: graph structure as obtained from Daphne
        Y: Y node values dict
        X: X node values dict
        R: Momentum
        new_X: Newly proposed X
        R_p: New momentum
        M_ivn: Inverse of mass covariance matrix

    Returns:
        Acceptance rate
    """

    return torch.exp(-1.0 * hmc_hamiltonian(graph, Y, new_X, R_p, M) +
                     hmc_hamiltonian(graph, Y, X, R, M))


def hmc(graph, S, T, eps, M):
    """
    Perform HMC sampling

    Args:
        graph: graph structure as obtained from Daphne
        S: Number of iterations to run HMC for
        T: Number of steps taken in the force field at each iteration
        eps: Length of each step
        M: Covariance of mass points

    Returns:
        List of sampled node values
    """

    observed_nodes = graph[1]['Y']

    # Need to obtain initial values for the nodes
    node_values = sample_from_joint(graph, only_latent_vars=True)

    # Set requires_grad to True for the nodes
    for node in node_values:
        node_values[node].requires_grad = True

    # Splitting nodes into X and Y here is helpful down the road
    X = {key: node_values[key] for key in node_values if key not in observed_nodes}
    Y = {key: node_values[key] for key in node_values if key in observed_nodes}

    # Initialize required distributions
    mv_normal_dist = dist.MultivariateNormal(torch.zeros(M.shape[0]), M)
    uniform_dist = dist.Uniform(0, 1)

    # List containing sampled nodes
    sampled_node_values = list()
    sampled_node_values.append(node_values)

    for s in range(S):

        # Sample an R
        R = mv_normal_dist.sample()

        # Need to make a copy of X as hmc_leap_frog would make changes to the original otherwise
        X_copy = copy.deepcopy(X)
        R_copy = R.clone()

        # Propose new R and X
        new_X, new_R = hmc_leap_frog(graph, Y, X_copy, R_copy, T, eps)

        # Calculate the acceptance rate
        acceptance_rate = hmc_accept(graph, Y, X, R, new_X, new_R, M)

        # Accept or reject new node values
        u = uniform_dist.sample()
        if u < acceptance_rate:
            X = new_X

        sampled_node_values.append({**X, **Y})

    return sampled_node_values[1:]


###########################################################################################################


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


def sample_from_joint(graph, only_latent_vars=False):
    """
    This function does ancestral sampling starting from the prior.

    Args:
        graph: graph structure as outputted by Daphne
        only_latent_vars: Indicates whether L is returned instead of final expression values

    Returns:
        Samples from the joint
    """


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
            if only_latent_vars:
                node_values[node] = evaluate_link_func(link_funcs[node][2], node_values, user_defns)
            else:
                # For now, we treat observe in a similar fashion to sample
                node_values[node] = evaluate_link_func(link_funcs[node][1], node_values, user_defns).sample()
        else:
            print('No support beyond sample and observe for random variable. Return None.')
            return None

    if only_latent_vars:
        return node_values
    else:
        # Evaluate the final return expression using the sampled values of random variables
        ret_val, _ = recursive_eval(return_expression, {}, node_values)

        return ret_val


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


def extract_output_samples(expression, sampled_node_values):
    """
    Generate the output expression samples

    Args:
        expression: output expression
        sampled_node_values: list of node values

    Returns:
        Samples from output expression
    """

    # List containing output expression values
    sampled_values = list()
    for v in sampled_node_values:
        sample, _ = recursive_eval(expression, {}, v)
        sampled_values.append(sample)

    sampled_values = torch.stack(sampled_values)
    return sampled_values


def extract_joint_log_prob(graph, sampled_node_values):
    """
    Generate joint log prob values

    Args:
        graph: graph as outputted by Daphne
        sampled_node_values: list of node values

    Returns:
        joint log values
    """

    user_defns = graph[0]
    link_funcs = graph[1]['P']

    # List containing output expression values
    log_probs = list()

    for sampled_node_value in sampled_node_values:
        log_prob = 0
        for node in sampled_node_value:
            if link_funcs[node][0] == 'sample*' or link_funcs[node][0] == 'observe*':
                log_prob += evaluate_link_func(link_funcs[node][1], sampled_node_value,
                                               user_defns).log_prob(sampled_node_value[node])
            else:
                continue

        log_probs.append(log_prob.item())

    return np.array(log_probs)


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


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)


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

    # run_deterministic_tests()
    # run_probabilistic_tests()

    for i in range(5, 6):
        graph = daphne(['graph', '-i', '../cpsc532w_hw/HW3/programs/{}.daphne'.format(i)])

        iterations = 1000

        # Compute program runtime
        # t_start = time.time()
        # sampled_node_values = gibbs(graph, iterations)
        # print('It took {} seconds for sampler with {} iterations.'.format((time.time() - t_start), iterations))
        #
        # samples = extract_output_samples(graph[2], sampled_node_values)
        #
        # print('The posterior expectation is {}. \n'.format(samples.mean(dim=0)))

        # t_start = time.time()
        # sampled_node_values = hmc(graph, iterations, 12, 0.11, torch.eye(len(graph[1]['V']) - len(graph[1]['Y'])))
        # print('It took {} seconds for sampler with {} iterations.'.format((time.time() - t_start), iterations))
        # samples = extract_output_samples(graph[2], sampled_node_values)
        # print('The posterior expectation is {}. \n'.format(samples.detach().mean(dim=1)))
