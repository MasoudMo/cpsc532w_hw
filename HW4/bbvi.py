import torch

from evaluation_based_sampling import recursive_eval, user_funcs
import wandb
import numpy as np
from daphne import daphne
from copy import deepcopy
import time
from distributions import Normal, Dirichlet, Categorical, Gamma


VISUALIZE = True


def elbo_gradients(G, logW, Q, use_baseline=True):
    """
    Calculate the ELBO gradients

    Args:
        G: list of dict of gradients with respect to v's params
        logW: List of logWs
        Q: Proposal dists
        use_baseline: Indicates whether b is computed or not

    Returns:
        gradient of ELBO
    """

    F = dict()

    # Extract nodes
    nodes = list()
    for g in G:
        nodes = nodes + list(g.keys())
    nodes = list(set(nodes))

    gradients = dict.fromkeys(nodes)
    G_new = dict()

    # Generate number of parameters dict
    param_nums = dict.fromkeys(nodes)
    for node in nodes:
        param_nums[node] = len(Q[node].Parameters())

    for node in nodes:
        for l in range(len(logW)):
            if node in G[l]:
                if node in F:
                    F[node].append(G[l][node] * logW[l])
                else:
                    F.update({node: [G[l][node] * logW[l]]})

                if node in G_new:
                    G_new[node].append(G[l][node])
                else:
                    G_new.update({node: [G[l][node]]})
            else:
                if node in F:
                    F[node].append(torch.tensor([0] * param_nums[node]))
                else:
                    F.update({node: torch.tensor([0] * param_nums[node])})
                G[l].update({node: torch.tensor([0] * param_nums[node])})
                if node in G_new:
                    G_new[node].append(torch.tensor([0] * param_nums[node]))
                else:
                    G_new.update({node: [torch.tensor([0] * param_nums[node])]})

        if use_baseline:

            # Solving an error where a 1 extra dim is added
            G_v_d = torch.stack(G_new[node]).squeeze().detach().numpy()
            F_v_d = torch.stack(F[node]).squeeze().detach().numpy()
            num = 0
            den = 0
            for d in range(G_v_d.shape[1]):
                num += np.cov(F_v_d[:, d], G_v_d[:, d])[1, 0]
                den += np.var(G_v_d[:, d])

            b_hat = num/den

            gradients[node] = np.reshape(np.sum(F_v_d - b_hat*G_v_d, axis=0) / len(logW), (param_nums[node], -1)).astype(np.float32)

        else:
            gradients[node] = np.reshape(torch.sum(torch.stack(F[node]), dim=0) / len(logW), (param_nums[node], -1))

    return gradients


def evaluate_link_func(expression, sigma, local_env, user_defns):
    """
    Evaluates the link functions

    Args:
        user_defns: Dictionary of user definitions
        sigma: state dict
        expression: Expression to be evaluated
        local_env: Dict containing variable bindings

    Returns:
        Expression return value
    """

    for func in user_defns.items():
        user_funcs.update({func[0]: {'args': func[1][1], 'body': func[1][2]}})

    ret, sigma = recursive_eval(expression, sigma, local_env)

    return ret, sigma

proposal_dists = {'normal': Normal,
                  'gamma': Gamma,
                  'discrete': Categorical,
                  'dirichlet': Dirichlet,
                  'uniform-continuous': Gamma}

def make_dist(dist_name):
    """
    Return proposal dist objects

    Args:
        dist_name: Name of dist

    Returns:
        dist object
    """

    if dist_name == 'discrete':
        return proposal_dists[dist_name](logits=torch.tensor([1.0, 1.0, 1.0]))
    elif dist_name == 'uniform-continuous':
        return proposal_dists[dist_name](concentration=torch.tensor(1.0), rate=torch.tensor(2.5))

    return proposal_dists[dist_name]()


def bbvi_evaluate(graph, sigma, sorted_node, prior_samples=None):
    """
    Evaluator wrapper for BBVI (For the simple case where no topological sort is required)

    Args:
        graph: graph structure as outputted by Daphne
        sigma: state dict
        sorted_node: topologically sorted nodes in the graph
        prior_samples: Prior samples used to initialize proposal dists

    Returns:
        Output expression sample and sigma
    """

    # Extract graph dics
    user_defns = graph[0]
    nodes = graph[1]['V']
    link_funcs = graph[1]['P']
    return_expression = graph[2]

    # Dictionary holding values for sampled nodes
    node_values = dict.fromkeys(nodes, None)

    for node in sorted_node:
        if link_funcs[node][0] == 'sample*':
            d, sigma = evaluate_link_func(link_funcs[node][1], sigma, node_values, user_defns)

            if node not in sigma['Q']:
                # sigma['Q'][node], _ = evaluate_link_func(link_funcs[node][1], sigma, prior_samples, user_defns)
                sigma['Q'][node] = make_dist(link_funcs[node][1][0])
                sigma['Q'][node] = sigma['Q'][node].make_copy_with_grads()

            # Generate a sample from q(x;nu)
            c = sigma['Q'][node].sample()
            node_values[node] = c.detach().clone()

            # Compute gradient with respect to q(v; nu)'s parameters
            sigma['G'].update({node: sigma['Q'][node].grad_log_prob(c).clone().detach()})

            # Compute the weight for specific variable
            logW_v = d.log_prob(c).detach() - sigma['Q'][node].log_prob(c).detach()

            # Add to weights
            sigma['logW'] += logW_v.detach()

        elif link_funcs[node][0] == 'observe*':
            d, sigma = evaluate_link_func(link_funcs[node][1], sigma, node_values, user_defns)
            c, sigma = evaluate_link_func(link_funcs[node][2], sigma, node_values, user_defns)
            node_values[node] = c

            # Add to weights
            sigma['logW'] += d.log_prob(c).detach()
        else:
            print('No support beyond sample and observe for random variable. Return None.')
            return None

    # Evaluate the final return expression using the sampled values of random variables
    ret_val, sigma = recursive_eval(return_expression, sigma, node_values)

    return ret_val, sigma


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


def sample_from_prior(graph, sorted_nodes):
    """
    This function does ancestral sampling starting from the prior.

    Args:
        graph: graph structure as outputted by Daphne
        sorted_nodes: topologically sorted nodes

    Returns:
        Samples from the prior
    """

    # Extract graph dics
    user_defns = graph[0]
    nodes = graph[1]['V']
    link_funcs = graph[1]['P']

    # Dictionary holding values for sampled nodes
    node_values = dict.fromkeys(nodes, None)

    for node in sorted_nodes:

        if link_funcs[node][0] == 'sample*':
            d, _ = evaluate_link_func(link_funcs[node][1], {}, node_values, user_defns)
            node_values[node] = d.sample()
        elif link_funcs[node][0] == 'observe*':
            node_values[node], _ = evaluate_link_func(link_funcs[node][2], {}, node_values, user_defns)
        else:
            print('No support beyond sample and observe for random variable. Return None.')
            return None

    return node_values


def optimizer_step(Q, gradients, lr=0.01, iteration=130):
    """

    Args:
        Q: Proposal dists
        gradients: Gradients of elbo
        lr: Learning rate
        iteration: Used if LR is iteration dependent

    Returns:
        New proposal dists
    """

    # lr = lr / torch.sqrt(torch.tensor(iteration + 1.0))

    for v in gradients:
        for i, param in enumerate(Q[v].Parameters()):
            param.data = param.data + lr * gradients[v][i]
            param.data = param.data.squeeze()
    return Q


def compute_identity_is_expectation(weighted_samples):
    """
    Computes posterior expection for identity function

    Args:
        weighted_samples: list of tuples of sample and weight pairs

    Returns:
        Expectation of posterior for identity function
    """

    # Extract the samples
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in weighted_samples]) * 1.0

    # Ensure correct shape for multiplication with W
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1)

    # Obtain log weights and exp them to obtain actual weights
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    weights = np.expand_dims(weights, axis=1)
    # weights = np.expand_dims(weights, axis=2)

    # Perform weighted average
    return np.sum((samples * weights) / np.sum(weights), axis=0)


def compute_identity_is_variance(weighted_samples, mean):
    """
    Computes posterior variance for identity function

    Args:
        weighted_samples: list of tuples of sample and weight pairs
        mean: mean of samples

    Returns:
        Variance of posterior for identity function
    """

    # Extract the samples
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in weighted_samples]) * 1.0

    # Ensure correct shape for multiplication with W
    if len(samples.shape) == 1:
        samples = np.expand_dims(samples, axis=1)

    # Obtain log weights and exp them to obtain actual weights
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    weights = np.expand_dims(weights, axis=1)

    # Perform weighted average
    return np.sum(((samples - mean)**2 * weights) / np.sum(weights), axis=0)


def compute_identity_is_dual_covariance(weighted_samples, mean):
    """
    Computes posterior co-variance for identity function

    Args:
        weighted_samples: list of tuples of sample and weight pairs
        mean: mean of samples

    Returns:
        Covariance of posterior for identity function
    """

    # Extract the samples
    samples_0 = np.array([np.array(weighted_sample[0][0]) for weighted_sample in weighted_samples])
    samples_1 = np.array([np.array(weighted_sample[0][1]) for weighted_sample in weighted_samples])

    samples_0 = np.expand_dims(samples_0, axis=1)
    samples_1 = np.expand_dims(samples_1, axis=1)

    # Obtain log weights and exp them to obtain actual weights
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    weights = np.expand_dims(weights, axis=1)

    variance00 = np.sum(((samples_0 - mean[0])**2 * weights) / np.sum(weights), axis=0)
    variance11 = np.sum(((samples_1 - mean[1])**2 * weights) / np.sum(weights), axis=0)
    variance01 = np.sum(((samples_0 - mean[0]) * (samples_1 - mean[1]) * weights) / np.sum(weights), axis=0)

    # Perform weighted average
    return np.array([[variance00, variance01], [variance01, variance11]])


def generate_weighted_samples(graph, path, iterations):
    """
    Generate weighted samples from trained proposal

    Args:
        graph: Graph as given by Daphne
        path: Path to saved proposal
        iterations: Number of samples to generate

    Returns:
        Generate weighted samples
    """

    # Initialize the elements in sigma
    sigma = {'logW': 0.0, 'Q': {}, 'G': {}}
    sigma['Q'] = torch.load(path)

    # To perform ancestral sampling, we should start from nodes with no dependence
    # This is achieved by performing topological sampling
    nodes = graph[1]['V']
    edges = graph[1]['A']
    sorted_nodes = top_sort(nodes, edges)

    weighted_samples = list()
    logW_t_l = list()
    # Optimization loop
    for i in range(iterations):
        r_t_l, sigma_t_l = bbvi_evaluate(graph=graph,
                                         sigma=sigma,
                                         sorted_node=sorted_nodes)
        logW_t_l.append(sigma_t_l['logW'])
        weighted_samples.append((r_t_l, sigma_t_l['logW']))
        sigma['logW'] = 0

    return weighted_samples


def bbvi(graph, T, L, path, lr):
    """
    Performs black box variational inference

    Args:
        graph: Graph structure as obtained from Daphne
        T: Number of optimization iterations
        L: Number of iterations for gradient estimation
        path: Path to save the optimized proposal to
        lr: learning rate

    Returns:
        Returns weighted samples from posterior
    """

    # Initialize visualization tool if needed
    if VISUALIZE:
        wandb.login(key='1251d7eb43e497238d7094b130c2fe2ab9286880')
        wandb.init(entity='madmas',
                   project='cpsc_532w_hw4',
                   config={'T': T,
                           'L': L,
                           'compiled_program': graph})


    # Initialize the elements in sigma
    sigma = {'logW': 0.0, 'Q': {}, 'G': {}}

    # To perform ancestral sampling, we should start from nodes with no dependence
    # This is achieved by performing topological sampling
    nodes = graph[1]['V']
    edges = graph[1]['A']
    sorted_nodes = top_sort(nodes, edges)

    # Generate the prior samples needed to initialize proposals
    prior_samples = sample_from_prior(graph, sorted_nodes)

    weighted_samples = list()
    # Optimization loop
    for t in range(T):

        G_t_l = list()
        logW_t_l = list()
        # Gradient estimation loop
        for l in range(L):
            r_t_l, sigma_t_l = bbvi_evaluate(graph=graph,
                                             sigma=sigma,
                                             sorted_node=sorted_nodes,
                                             prior_samples=prior_samples)

            G_t_l.append(deepcopy(sigma_t_l['G']))
            logW_t_l.append(sigma_t_l['logW'])
            weighted_samples.append((r_t_l, sigma_t_l['logW']))
            sigma['logW'] = 0

        # Only perform this if weights are valid
        gradients = elbo_gradients(G_t_l, logW_t_l, sigma['Q'])
        sigma['Q'] = optimizer_step(sigma['Q'], gradients, lr=lr, iteration=t)

        elbo = sum(logW_t_l)/L
        # mean = compute_identity_is_expectation(weighted_samples)
        print('Intermediate ELBO: {}'.format(elbo))
        # print('Intermediate Mean: {}'.format(mean))

        if VISUALIZE:
            wandb.log({'ELBO': elbo})

        # Save the learned distributions
        torch.save(sigma['Q'], path)

    if VISUALIZE:
        wandb.finish()

    return


if __name__ == '__main__':
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW4/programs/1.daphne'])

    T = 500
    L = 1000

    t_start = time.time()
    bbvi(graph, T, L)
    print('It took {} seconds for sampler with T:{} L:{} iterations.'.format((time.time() - t_start), T, L))

    weighted_samples = generate_weighted_samples(graph, './dist.pt', 100000)
    expectation = compute_identity_is_expectation(weighted_samples)
    variance = compute_identity_is_variance(weighted_samples, expectation)
