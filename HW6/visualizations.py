import numpy as np

from evaluator import evaluate
from daphne import daphne
import matplotlib.pyplot as plt
import time
from smc import SMC
import torch
import seaborn as sns


if __name__ == '__main__':

    ##################################### Program 1 ######################################
    exp = daphne(['desugar-hoppl-cps', '-i', '../cpsc532w_hw/HW6/programs/1.daphne'])

    plt.figure()
    for n_particles in [1, 10, 100, 1000, 10000, 100000]:

        # Generate samples from the prior
        t_start = time.time()
        logZ, particles = SMC(n_particles, exp)
        t_end = time.time()

        print('logZ: ', logZ)

        values = torch.stack(particles).detach().numpy()

        # Compute the expected mean and variance
        mean = np.mean(values)
        var = np.var(values)

        # Print the results
        print('Program 1: It took {} seconds with {} particles.'.format((t_end - t_start), n_particles))
        print('The posterior expectation value is: {}'.format(mean))
        print('The posterior variance value is: {}'.format(var))

        # Plot the histogram
        plt.hist(values, bins=30)
        plt.title('Program 1 Histogram with {} particles'.format(n_particles))
        plt.xlabel('Sample Value')
        plt.ylabel('Frequency')

        plt.show()
        plt.clf()
    ######################################################################################

    ##################################### Program 2 ######################################
    exp = daphne(['desugar-hoppl-cps', '-i', '../cpsc532w_hw/HW6/programs/2.daphne'])

    plt.figure()
    for i, n_particles in enumerate([1, 10, 100, 1000, 10000, 100000]):

        # Generate samples from the prior
        t_start = time.time()
        logZ, particles = SMC(n_particles, exp)
        t_end = time.time()

        print('logZ: ', logZ)

        values = torch.stack(particles).detach().numpy()

        # Compute the expected mean and variance
        mean = np.mean(values)
        var = np.var(values)

        # Print the results
        print('Program 2: It took {} seconds with {} particles.'.format((t_end - t_start), n_particles))
        print('The posterior expectation value is: {}'.format(mean))
        print('The posterior variance value is: {}'.format(var))

        # Plot the histogram
        plt.hist(values, bins=30)
        plt.title('Program 2 Histogram with {} particles'.format(n_particles))
        plt.xlabel('Sample Value')
        plt.ylabel('Frequency')

        plt.show()
        plt.clf()

    # ######################################################################################
    #
    # ##################################### Program 3 ######################################
    exp = daphne(['desugar-hoppl-cps', '-i', '../cpsc532w_hw/HW6/programs/3.daphne'])

    num_samples = 100000
    samples = list()

    plt.figure()
    for i, n_particles in enumerate([1, 10, 100, 1000, 10000, 100000]):

        # Generate samples from the prior
        t_start = time.time()
        logZ, particles = SMC(n_particles, exp)
        t_end = time.time()

        print('logZ: ', logZ)

        values = torch.stack(particles).detach().numpy()

        # Compute the expected mean and variance
        mean = np.mean(values, axis=0)
        var = np.var(values, axis=0)

        # Print the results
        print('Program 3: It took {} seconds with {} particles.'.format((t_end - t_start), n_particles))
        print('The posterior expectation value for each dim is: {}'.format(mean))
        print('The posterior variance value for each dim is: {}'.format(var))

        # Plot heatmaps
        sns.heatmap(np.expand_dims(mean, axis=1), annot=True, fmt='g')
        plt.show()
        plt.clf()
        sns.heatmap(np.expand_dims(var, axis=1), annot=True, fmt='g')
        plt.show()
        plt.clf()

        # Plot the histograms
        fig0, axs0 = plt.subplots(6, 3)
        fig0.tight_layout(pad=0.5)

        for dim in range(17):
            axs0[int(dim / 3), int(dim % 3)].hist(values[:, dim], bins=30)
            axs0[int(dim / 3), int(dim % 3)].set_title('Histogram for dim {}'.format(dim), fontsize=10)
            axs0[int(dim / 3), int(dim % 3)].set_xlabel('Sample Value Bins')
            axs0[int(dim / 3), int(dim % 3)].set_ylabel('Frequency')

        plt.show()
        plt.clf()

    ##################################### Program 2 ######################################
    exp = daphne(['desugar-hoppl-cps', '-i', '../cpsc532w_hw/HW6/programs/4.daphne'])

    plt.figure()
    for i, n_particles in enumerate([1, 10, 100, 1000, 10000, 100000]):

        # Generate samples from the prior
        t_start = time.time()
        logZ, particles = SMC(n_particles, exp)
        t_end = time.time()

        print('logZ: ', logZ)

        values = torch.stack(particles).detach().numpy()

        # Compute the expected mean and variance
        mean = np.mean(values)
        var = np.var(values)

        # Print the results
        print('Program 4: It took {} seconds with {} particles.'.format((t_end - t_start), n_particles))
        print('The posterior expectation value is: {}'.format(mean))
        print('The posterior variance value is: {}'.format(var))

        # Plot the histogram
        plt.hist(values, bins=30)
        plt.title('Program 4 Histogram with {} particles'.format(n_particles))
        plt.xlabel('Sample Value')
        plt.ylabel('Frequency')

        plt.show()
        plt.clf()

    # ######################################################################################