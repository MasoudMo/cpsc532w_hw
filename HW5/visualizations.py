import numpy as np

from evaluator import evaluate
from daphne import daphne
import matplotlib.pyplot as plt
import time
import threading
import seaborn as sns

# Set recursion limit as I got an error for program 1 (solution from:
# https://stackoverflow.com/questions/8177073/python-maximum-recursion-depth-exceeded)
import sys


def program_1():
    exp = daphne(['desugar-hoppl', '-i', '../cpsc532w_hw/HW5/programs/1.daphne'])

    num_samples = 50000
    samples = list()

    # Generate samples from the prior
    t_start = time.time()
    for i in range(num_samples):
        samples.append(evaluate(exp).detach().item())
    t_end = time.time()

    # Compute the expected mean and variance
    mean = np.mean(samples)
    var = np.var(samples)

    # Print the results
    print('Program 1: It took {} seconds with {} samples.'.format((t_end - t_start), num_samples))
    print('The prior expectation value is: {}'.format(mean))
    print('The prior variance value is: {}'.format(var))

    # Plot the histogram
    plt.figure(0)
    plt.hist(samples, bins=30)
    plt.title('Program 1 Histogram of Geometrical Distribution with P=0.01')
    plt.xlabel('Sample Value')
    plt.ylabel('Frequency')

    plt.show()


if __name__ == '__main__':

    sys.setrecursionlimit(7000)

    ##################################### Program 1 ######################################
    threading.stack_size(200000000)
    thread = threading.Thread(target=program_1)
    thread.start()
    thread.join()

    ######################################################################################

    ##################################### Program 2 ######################################
    exp = daphne(['desugar-hoppl', '-i', '../cpsc532w_hw/HW5/programs/2.daphne'])

    num_samples = 100000
    samples = list()

    # Generate samples from the prior
    t_start = time.time()
    for i in range(num_samples):
        samples.append(evaluate(exp).detach().item())
    t_end = time.time()

    # Compute the expected mean and variance
    mean = np.mean(samples)
    var = np.var(samples)

    # Print the results
    print('Program 2: It took {} seconds with {} samples.'.format((t_end - t_start), num_samples))
    print('The prior expectation value is: {}'.format(mean))
    print('The prior variance value is: {}'.format(var))

    # Plot the histogram
    plt.figure(1)
    plt.hist(samples, bins=30)
    plt.title('Program 2 Histogram of mu')
    plt.xlabel('Sample Value')
    plt.ylabel('Frequency')

    plt.show()

    ######################################################################################

    ##################################### Program 3 ######################################
    exp = daphne(['desugar-hoppl', '-i', '../cpsc532w_hw/HW5/programs/3.daphne'])

    num_samples = 100000
    samples = list()

    # Generate samples from the prior
    t_start = time.time()
    for i in range(num_samples):
        samples.append(evaluate(exp).detach().numpy())
    t_end = time.time()
    samples = np.array(samples)

    # Compute the expected mean and variance
    mean = np.mean(samples, axis=0)
    var = np.var(samples, axis=0)

    # Print the results
    print('Program 3: It took {} seconds with {} samples.'.format((t_end - t_start), num_samples))
    print('The prior expectation value for each dim is: {}'.format(mean))
    print('The prior variance value for each dim is: {}'.format(var))

    # Plot heatmaps
    plt.figure(2)
    sns.heatmap(np.expand_dims(mean, axis=1), annot=True, fmt='g')
    plt.figure(3)
    sns.heatmap(np.expand_dims(var, axis=1), annot=True, fmt='g')

    # Plot the histograms
    fig0, axs0 = plt.subplots(6, 3)
    fig0.tight_layout(pad=0.5)

    for dim in range(17):
        axs0[int(dim / 3), int(dim % 3)].hist(samples[:, dim], bins=30)
        axs0[int(dim / 3), int(dim % 3)].set_title('Histogram for dim {}'.format(dim), fontsize=10)
        axs0[int(dim / 3), int(dim % 3)].set_xlabel('Sample Value Bins')
        axs0[int(dim / 3), int(dim % 3)].set_ylabel('Frequency')

    plt.show()
