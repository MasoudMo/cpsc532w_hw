import numpy as np

from graph_based_sampling import gibbs, extract_output_samples, extract_joint_log_prob, hmc
from evaluation_based_sampling import evaluate_likelihood_weighting, compute_identity_is_variance, \
    compute_identity_is_expectation, compute_identity_is_dual_covariance
from daphne import daphne
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time

if __name__ == '__main__':

    # For each case, I use both evaluation based and graph based sampling


    ############################################################################
    # Program 1
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW3/programs/1.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW3/programs/1.daphne'])

    # Number of iterations
    iterations = 2000000

    # Perfrom IS likelihood sampling
    t_start = time.time()
    weighted_samples = evaluate_likelihood_weighting(ast, iterations)
    print('Program 1 IS sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    mean = compute_identity_is_expectation(weighted_samples)
    variance = compute_identity_is_variance(weighted_samples, mean)
    print('Program 1 IS sampler: The posterior mean is {}'.format(mean))
    print('Program 1 IS sampler: The variance is {}'.format(variance))

    # Plot the histogram
    plt.figure(0)
    plt.title('Program 1 - Importance Sampling - Histogram')
    samples = np.array([weighted_sample[0] for weighted_sample in weighted_samples])
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    plt.hist(samples, bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Perfrom Gibbs Sampling
    iterations = 500000
    t_start = time.time()
    sampled_node_values = gibbs(graph, iterations)
    print('Program 1 Gibbs sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values)
    print('Program 1 Gibbs sampler: The posterior mean is {}'.format(samples.mean()))
    print('Program 1 Gibbs sampler: The variance is {}'.format(samples.std() ** 2))

    # Plot the histogram
    plt.figure(1)
    plt.title('Program 1 - Gibbs Sampling - Histogram')
    plt.hist(samples.detach().numpy(), bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Plot the trace
    plt.figure(2)
    plt.title('Program 1 - Gibbs Sampling - Trace')
    plt.plot(range(len(samples)), samples)
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    # Plot the joint log probs
    joint_log_probs = extract_joint_log_prob(graph, sampled_node_values)
    plt.figure(3)
    plt.title('Program 1 - Gibbs Sampling - Joint Probs')
    plt.plot(range(len(joint_log_probs)), joint_log_probs)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Probability')

    # Perfrom HMC Sampling
    iterations = 60000
    t_start = time.time()
    sampled_node_values = hmc(graph, iterations, 12, 0.11, torch.eye(len(graph[1]['V']) - len(graph[1]['Y'])))
    print('Program 1 HMC sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values)
    print('Program 1 HMC sampler: The posterior mean is {}'.format(samples.mean()))
    print('Program 1 HMC sampler: The variance is {}'.format(samples.std() ** 2))

    # Plot the histogram
    plt.figure(4)
    plt.title('Program 1 - HMC Sampling - Histogram')
    plt.hist(samples.detach().numpy(), bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Plot the trace
    plt.figure(5)
    plt.title('Program 1 - HMC Sampling - Trace')
    plt.plot(range(len(samples.detach().numpy())), samples.detach().numpy())
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    # Plot the joint log probs
    joint_log_probs = extract_joint_log_prob(graph, sampled_node_values)
    plt.figure(6)
    plt.title('Program 1 - HMC Sampling - Joint Probs')
    plt.plot(range(len(joint_log_probs)), joint_log_probs)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Probability')

    plt.show()
    ############################################################################

    # ############################################################################
    # Program 2

    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW3/programs/2.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW3/programs/2.daphne'])

    # Number of iterations
    iterations = 550000

    # Perfrom IS likelihood sampling
    t_start = time.time()
    weighted_samples = evaluate_likelihood_weighting(ast, iterations)
    print('Program 2 IS sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    weighted_samples_0 = np.array([(weighted_sample[0][0], weighted_sample[1]) for weighted_sample in weighted_samples])
    weighted_samples_1 = np.array([(weighted_sample[0][1], weighted_sample[1]) for weighted_sample in weighted_samples])
    mean = compute_identity_is_expectation(weighted_samples)
    cov = compute_identity_is_dual_covariance(weighted_samples, mean)
    plt.figure(7)
    sns.heatmap(cov[:, :, 0], annot=True, fmt='g')
    print('Program 2 IS sampler: The posterior means for slope and bias are {}'.format(mean))

    # Plot the histograms
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in weighted_samples])
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)

    plt.figure(8)
    plt.title('Program 2 - Importance Sampling - Histogram - Slope')
    plt.hist(samples[:, 0], bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.figure(9)
    plt.title('Program 2 - Importance Sampling - Histogram - Bias')
    plt.hist(samples[:, 1], bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Perfrom Gibbs Sampling
    iterations = 100000
    t_start = time.time()
    sampled_node_values = gibbs(graph, iterations)
    print('Program 2 Gibbs sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values)
    cov = np.cov(samples.detach().numpy(), rowvar=False)
    plt.figure(10)
    sns.heatmap(cov, annot=True, fmt='g')
    print('Program 2 Gibbs sampler: The posterior means for slope and bias are {}'.format(samples.mean(dim=0)))

    # Plot the histograms
    plt.figure(11)
    plt.title('Program 2 - Gibbs Sampling - Histogram - Slope')
    plt.hist(samples.detach().numpy()[:, 0], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.figure(12)
    plt.title('Program 2 - Gibbs Sampling - Histogram - Bias')
    plt.hist(samples.detach().numpy()[:, 1], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Plot the traces
    plt.figure(13)
    plt.title('Program 2 - Gibbs Sampling - Trace - Slope')
    plt.plot(range(len(samples)), np.array(samples)[:, 0])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    plt.figure(14)
    plt.title('Program 2 - Gibbs Sampling - Trace - Bias')
    plt.plot(range(len(samples)), np.array(samples)[:, 1])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    # Plot the joint log probs
    joint_log_probs = extract_joint_log_prob(graph, sampled_node_values)
    plt.figure(15)
    plt.title('Program 2 - Gibbs Sampling - Joint Probs')
    plt.plot(range(len(joint_log_probs)), joint_log_probs)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Probability')

    # Perfrom HMC Sampling
    iterations = 25000
    t_start = time.time()
    sampled_node_values = hmc(graph, iterations, 12, 0.11, torch.eye(len(graph[1]['V']) - len(graph[1]['Y'])))
    print('Program 2 HMC sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values)
    cov = np.cov(samples.detach().numpy(), rowvar=False)
    plt.figure(16)
    sns.heatmap(cov, annot=True, fmt='g')
    print('Program 2 HMC sampler: The posterior means for slope and bias are {}'.format(samples.mean(dim=0)))

    # Plot the histograms
    plt.figure(17)
    plt.title('Program 2 - HMC Sampling - Histogram - Slope')
    plt.hist(samples.detach().numpy()[:, 0], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.figure(18)
    plt.title('Program 2 - HMC Sampling - Histogram - Bias')
    plt.hist(samples.detach().numpy()[:, 1], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Plot the traces
    plt.figure(19)
    plt.title('Program 2 - HMC Sampling - Trace - Slope')
    plt.plot(range(len(samples)), samples.detach().numpy()[:, 0])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    plt.figure(20)
    plt.title('Program 2 - HMC Sampling - Trace - Bias')
    plt.plot(range(len(samples)), samples.detach().numpy()[:, 1])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    # Plot the joint log probs
    joint_log_probs = extract_joint_log_prob(graph, sampled_node_values)
    plt.figure(21)
    plt.title('Program 2 - HMC Sampling - Joint Probs')
    plt.plot(range(len(joint_log_probs)), joint_log_probs)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Probability')

    plt.show()

    # # ############################################################################
    # Program 3
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW3/programs/3.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW3/programs/3.daphne'])

    # Number of iterations
    iterations = 300000

    # Perfrom IS likelihood sampling
    t_start = time.time()
    weighted_samples = evaluate_likelihood_weighting(ast, iterations)
    print('Program 3 IS sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    mean = compute_identity_is_expectation(weighted_samples)
    variance = compute_identity_is_variance(weighted_samples, mean)
    print('Program 3 IS sampler: The posterior mean is {}'.format(mean))
    print('Program 3 IS sampler: The variance is {}'.format(variance))

    # Plot the histogram
    plt.figure(22)
    plt.title('Program 3 - Importance Sampling - Histogram')
    samples = np.array([weighted_sample[0] for weighted_sample in weighted_samples]) * 1.0
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    plt.hist(samples, bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Perfrom Gibbs Sampling
    iterations = 16000
    t_start = time.time()
    sampled_node_values = gibbs(graph, iterations)
    print('Program 3 Gibbs sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values) * 1.0
    print('Program 3 Gibbs sampler: The posterior mean is {}'.format(samples.mean()))
    print('Program 3 Gibbs sampler: The variance is {}'.format(samples.std() ** 2))

    # Plot the histogram
    plt.figure(23)
    plt.title('Program 3 - Gibbs Sampling - Histogram')
    plt.hist(samples.detach().numpy(), bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')
    #
    plt.show()

    # ############################################################################
    # Program 4

    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW3/programs/4.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW3/programs/4.daphne'])

    # Number of iterations
    iterations = 1000000

    # Perfrom IS likelihood sampling
    t_start = time.time()
    weighted_samples = evaluate_likelihood_weighting(ast, iterations)
    print('Program 4 IS sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    mean = compute_identity_is_expectation(weighted_samples)
    variance = compute_identity_is_variance(weighted_samples, mean)
    print('Program 4 IS sampler: The posterior mean is {}'.format(mean))
    print('Program 4 IS sampler: The variance is {}'.format(variance))

    # Plot the histogram
    plt.figure(24)
    plt.title('Program 4 - Importance Sampling - Histogram')
    samples = np.array([weighted_sample[0] for weighted_sample in weighted_samples])
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    plt.hist(samples, bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Perfrom Gibbs Sampling
    iterations = 150000
    t_start = time.time()
    sampled_node_values = gibbs(graph, iterations)
    print('Program 4 Gibbs sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values) * 1.0
    print('Program 4 Gibbs sampler: The posterior mean is {}'.format(samples.mean()))
    print('Program 4 Gibbs sampler: The variance is {}'.format(samples.std() ** 2))

    # Plot the histogram
    plt.figure(25)
    plt.title('Program 4 - Gibbs Sampling - Histogram')
    plt.hist(samples.detach().numpy(), bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.show()

    ############################################################################
    # Program 2

    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW3/programs/5.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW3/programs/5.daphne'])

    # Number of iterations
    iterations = 1000000

    # Perfrom IS likelihood sampling
    t_start = time.time()
    weighted_samples = evaluate_likelihood_weighting(ast, iterations)
    print('Program 5 IS sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    weighted_samples_0 = np.array([(weighted_sample[0][0], weighted_sample[1]) for weighted_sample in weighted_samples])
    weighted_samples_1 = np.array([(weighted_sample[0][1], weighted_sample[1]) for weighted_sample in weighted_samples])
    mean = compute_identity_is_expectation(weighted_samples)
    var0 = compute_identity_is_variance(weighted_samples_0, mean[0])
    var1 = compute_identity_is_variance(weighted_samples_1, mean[1])
    print('Program 5 IS sampler: The posterior means for x and y are {}'.format(mean))
    print('Program 5 IS sampler: The variance for x and y is: {}, {}'.format(var0, var1))

    # Plot the histograms
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in weighted_samples])
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)

    plt.figure(26)
    plt.title('Program 5 - Importance Sampling - Histogram - X')
    plt.hist(samples[:, 0], bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.figure(27)
    plt.title('Program 5 - Importance Sampling - Histogram - Y')
    plt.hist(samples[:, 1], bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Perfrom Gibbs Sampling
    iterations = 500000
    t_start = time.time()
    sampled_node_values = gibbs(graph, iterations)
    print('Program 5 Gibbs sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values)
    print('Program 5 Gibbs sampler: The posterior means for x and y are {}'.format(samples.mean(dim=0)))
    print('Program 5 Gibbs sampler: The posterior variances for x and y are {}'.format(samples.std(dim=0) ** 2))

    # Plot the histograms
    plt.figure(28)
    plt.title('Program 5 - Gibbs Sampling - Histogram - X')
    plt.hist(samples.detach().numpy()[:, 0], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.figure(29)
    plt.title('Program 5 - Gibbs Sampling - Histogram - Y')
    plt.hist(samples.detach().numpy()[:, 1], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Plot the traces
    plt.figure(30)
    plt.title('Program 2 - Gibbs Sampling - Trace - X')
    plt.plot(range(len(samples)), np.array(samples)[:, 0])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    plt.figure(31)
    plt.title('Program 2 - Gibbs Sampling - Trace - Y')
    plt.plot(range(len(samples)), np.array(samples)[:, 1])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    # Plot the joint log probs
    joint_log_probs = extract_joint_log_prob(graph, sampled_node_values)
    plt.figure(32)
    plt.title('Program 5 - Gibbs Sampling - Joint Probs')
    plt.plot(range(len(joint_log_probs)), joint_log_probs)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Probability')

    # Perfrom HMC Sampling
    iterations = 200000
    t_start = time.time()
    sampled_node_values = hmc(graph, iterations, 5, 0.05, torch.eye(len(graph[1]['V']) - len(graph[1]['Y'])))
    print('Program 5 HMC sampler: It took {} seconds with {} iterations.'.format((time.time() - t_start), iterations))

    # Compute its variance and mean
    samples = extract_output_samples(graph[2], sampled_node_values)
    print('Program 5 HMC sampler: The posterior means for x and y are {}'.format(samples.mean(dim=0).detach().numpy()))
    print('Program 5 HMC sampler: The posterior variances for x and y are {}'.format(samples.std(dim=0).detach().numpy() ** 2))

    # Plot the histograms
    plt.figure(33)
    plt.title('Program 5 - HMC Sampling - Histogram - X')
    plt.hist(samples.detach().numpy()[:, 0], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.figure(34)
    plt.title('Program 5 - HMC Sampling - Histogram - Y')
    plt.hist(samples.detach().numpy()[:, 1], bins=60)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    # Plot the traces
    plt.figure(35)
    plt.title('Program 5 - HMC Sampling - Trace - X')
    plt.plot(range(len(samples)), samples.detach().numpy()[:, 0])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    plt.figure(36)
    plt.title('Program 2 - HMC Sampling - Trace - Y')
    plt.plot(range(len(samples)), samples.detach().numpy()[:, 1])
    plt.xlabel('Iteration')
    plt.ylabel('Output Value')

    # Plot the joint log probs
    joint_log_probs = extract_joint_log_prob(graph, sampled_node_values)
    plt.figure(37)
    plt.title('Program 5 - HMC Sampling - Joint Probs')
    plt.plot(range(len(joint_log_probs)), joint_log_probs)
    plt.xlabel('Iteration')
    plt.ylabel('Joint Probability')

    plt.show()

    ############################################################################

