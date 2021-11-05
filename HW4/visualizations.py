import numpy as np

from bbvi import bbvi, generate_weighted_samples, compute_identity_is_expectation, compute_identity_is_variance, \
    compute_identity_is_dual_covariance
from daphne import daphne
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time

if __name__ == '__main__':

    # For each case, I use both evaluation based and graph based sampling


    ############################################################################
    # Program 1
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW4/programs/1.daphne'])

    # Number of iterations
    iterations = 100000
    lr = 0.1
    T = 100
    L = 500
    path = './program1_dist.pt'

    # Perfrom BBVI5
    t_start = time.time()
    # bbvi(graph, T=T, L=L, path=path, lr=lr)
    print('Program 1: It took {} seconds for T:{} and L:{}'.format((time.time() - t_start), T, L))

    # Compute its variance and mean
    weighted_samples = generate_weighted_samples(graph, path=path, iterations=iterations)
    mean = compute_identity_is_expectation(weighted_samples)
    variance = compute_identity_is_variance(weighted_samples, mean)
    print('Program 1: The posterior mean is {}'.format(mean))
    print('Program 1: The variance is {}'.format(variance))

    # Plot the histogram
    plt.figure(0)
    plt.title('Program 1 - Histogram')
    samples = np.array([weighted_sample[0] for weighted_sample in weighted_samples])
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    plt.hist(samples, bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.show()

    ###########################################################################

    ###############################Program 2###################################
    # Program 2
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW4/programs/2.daphne'])

    # Number of iterations
    iterations = 100000
    lr = 0.01
    T = 500
    L = 500
    path = './program2_dist.pt'

    # Perfrom BBVI
    t_start = time.time()
    bbvi(graph, T=T, L=L, path=path, lr=lr)
    print('Program 2: It took {} seconds for T:{} and L:{}'.format((time.time() - t_start), T, L))

    # Compute its variance and mean
    weighted_samples = generate_weighted_samples(graph, path=path, iterations=iterations)
    weighted_samples_0 = np.array([(weighted_sample[0][0], weighted_sample[1]) for weighted_sample in weighted_samples])
    weighted_samples_1 = np.array([(weighted_sample[0][1], weighted_sample[1]) for weighted_sample in weighted_samples])
    mean = compute_identity_is_expectation(weighted_samples)
    cov = compute_identity_is_dual_covariance(weighted_samples, mean)
    plt.figure(1)
    sns.heatmap(cov[:, :, 0], annot=True, fmt='g')
    print('Program 2: The posterior means for slope and bias are {}'.format(mean))

    # Plot the histograms
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in weighted_samples])
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)

    plt.figure(2)
    plt.title('Program 2 - Histogram - Slope')
    plt.hist(samples[:, 0], bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.figure(3)
    plt.title('Program 2 - Histogram - Bias')
    plt.hist(samples[:, 1], bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.show()

    ###########################################################################

    ###############################Program 3###################################
    # Program 3
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW3/programs/3.daphne'])

    # Number of iterations
    iterations = 10000
    lr = 0.01
    T = 500
    L = 500
    path = './program3_dist.pt'

    # Perfrom BBVI
    t_start = time.time()
    bbvi(graph, T=T, L=L, path=path, lr=lr)
    print('Program 3: It took {} seconds for T:{} and L:{}'.format((time.time() - t_start), T, L))

    # Compute its variance and mean
    weighted_samples = generate_weighted_samples(graph, path=path, iterations=iterations)
    mean = compute_identity_is_expectation(weighted_samples)
    variance = compute_identity_is_variance(weighted_samples, mean)
    print('Program 3: The posterior mean is {}'.format(mean))
    print('Program 3: The variance is {}'.format(variance))

    # Plot the histogram
    plt.figure(4)
    plt.title('Program 3 - Histogram')
    samples = np.array([weighted_sample[0] for weighted_sample in weighted_samples]) * 1.0
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    plt.hist(samples, bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.show()

    ###########################################################################

    ###############################Program 4###################################
    # Program 4
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW4/programs/4.daphne'])

    # Number of iterations
    iterations = 100000
    lr = 0.01
    T = 500
    L = 150
    path = './program4_dist_3.pt'

    # Perfrom BBVI
    t_start = time.time()
    bbvi(graph, T=T, L=L, path=path, lr=lr)
    print('Program 4: It took {} seconds for T:{} and L:{}'.format((time.time() - t_start), T, L))

    W_0 = list()
    b_0 = list()
    W_1 = list()
    b_1 = list()
    weights = list()
    weighted_samples = generate_weighted_samples(graph, path, iterations)
    for weighted_sample in weighted_samples:
        weights.append(weighted_sample[1])
        W_0.append((weighted_sample[0][0], weighted_sample[1]))
        b_0.append((weighted_sample[0][1], weighted_sample[1]))
        W_1.append((weighted_sample[0][2], weighted_sample[1]))
        b_1.append((weighted_sample[0][3], weighted_sample[1]))

    # Extract Weights
    log_weights = np.array([weighted_sample[1] for weighted_sample in W_0])
    weights = np.exp(log_weights.astype(np.double))

    # W0 Visualizations
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in W_0]) * 1.0
    w0_mean = np.average(samples, weights=weights, axis=0)
    w0_std = np.average((samples-w0_mean)**2, weights=weights, axis=0)

    plt.figure(5)
    sns.heatmap(w0_mean, annot=True)
    plt.title('Program 4 - Heatmap for W0 Mean')

    plt.figure(6)
    sns.heatmap(w0_std, annot=True)
    plt.title('Program 4 - Heatmap for W0 Variance')

    # b0 Visualizations
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in b_0]) * 1.0
    b0_mean = np.average(samples, weights=weights, axis=0)
    b0_std = np.average((samples-b0_mean)**2, weights=weights, axis=0)

    plt.figure(7)
    sns.heatmap(b0_mean, annot=True)
    plt.title('Program 4 - Heatmap for b0 Mean')

    plt.figure(8)
    sns.heatmap(b0_std, annot=True)
    plt.title('Program 4 - Heatmap for b0 Variance')

    # W1 Visualizations
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in W_1]) * 1.0
    w1_mean = np.average(samples, weights=weights, axis=0)
    w1_std = np.average((samples-w1_mean)**2, weights=weights, axis=0)

    plt.figure(9)
    sns.heatmap(w1_mean, annot=True)
    plt.title('Program 4 - Heatmap for W1 Mean')

    plt.figure(10)
    sns.heatmap(w1_std, annot=True)
    plt.title('Program 4 - Heatmap for W1 Variance')

    # b1 Visualizations
    samples = np.array([np.array(weighted_sample[0]) for weighted_sample in b_1]) * 1.0
    b1_mean = np.average(samples, weights=weights, axis=0)
    b1_std = np.average((samples-b1_mean)**2, weights=weights, axis=0)

    plt.figure(11)
    sns.heatmap(b1_mean, annot=True)
    plt.title('Program 4 - Heatmap for b1 Mean')

    plt.figure(12)
    sns.heatmap(b1_std, annot=True)
    plt.title('Program 4 - Heatmap for b1 Variance')

    plt.show()

###########################################################################

###############################Program 5###################################
    # Program 5
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW4/programs/5.daphne'])

    # Number of iterations
    iterations = 100000
    lr = 0.001
    T = 10000
    L = 100
    path = './program5_dist.pt'

    # Perfrom BBVI
    t_start = time.time()
    bbvi(graph, T=T, L=L, path=path, lr=lr)
    print('Program 5: It took {} seconds for T:{} and L:{}'.format((time.time() - t_start), T, L))

    # Compute its variance and mean
    weighted_samples = generate_weighted_samples(graph, path=path, iterations=iterations)
    mean = compute_identity_is_expectation(weighted_samples)
    variance = compute_identity_is_variance(weighted_samples, mean)
    print('Program 5: The posterior mean is {}'.format(mean))
    print('Program 5: The variance is {}'.format(variance))

    # Plot the histogram
    plt.figure(13)
    plt.title('Program 5 - Histogram')
    samples = np.array([weighted_sample[0] for weighted_sample in weighted_samples])
    log_weights = np.array([weighted_sample[1] for weighted_sample in weighted_samples])
    weights = np.exp(log_weights)
    plt.hist(samples, bins=60, weights=weights)
    plt.xlabel('Sample Value Bins')
    plt.ylabel('Frequency')

    plt.show()

    ############################################################################
