import numpy as np

from graph_based_sampling import sample_from_joint
from evaluation_based_sampling import evaluate_program
from daphne import daphne
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    # For each case, I use both evaluation based and graph based sampling


    ############################################################################
    # Program 1
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW2/programs/1.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/1.daphne'])

    gb_samples = list()
    ev_samples = list()

    for j in range(1000):
        gb_samples.append(float(sample_from_joint(graph).numpy()))
        ev_samples.append(float(evaluate_program(ast).numpy()))

    print("Program 1: Marginal Expectation for Eval Based with 1000 samples = {}".format(sum(gb_samples)/1000))
    print("Program 1: Marginal Expectation for Graph Based with 1000 samples = {} \n \n".format(sum(ev_samples)/1000))

    fig, axs = plt.subplots(1, 2)

    axs[0].hist(gb_samples, bins=20)
    axs[0].set_title('1000 Graph Based Samples')
    axs[0].set_xlabel('Sample Value Bins')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(ev_samples, bins=20)
    axs[1].set_title('1000 Evaluation Based Samples')
    axs[1].set_xlabel('Sample Value Bins')
    axs[1].set_ylabel('Frequency')

    plt.show()
    ############################################################################

    ############################################################################
    # Program 2
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW2/programs/2.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/2.daphne'])

    gb_samples = np.zeros((2, 1000))
    ev_samples = np.zeros((2, 1000))

    for j in range(1000):
        gb_samples[:, j] = sample_from_joint(graph).numpy()
        ev_samples[:, j] = evaluate_program(ast).numpy()

    print("Program 2: Marginal Expectation for dim 0 for Graph Based with 1000 samples = {}".format(np.sum(
        gb_samples[0, :]) / 1000))

    print("Program 2: Marginal Expectation for dim 1 for Graph Based with 1000 samples = {}".format(np.sum(
        gb_samples[1, :]) / 1000))

    print("Program 2: Marginal Expectation for dim 0 for Eval Based with 1000 samples = {}".format(np.sum(
        ev_samples[0, :]) / 1000))

    print("Program 2: Marginal Expectation for dim 1 for Eval Based with 1000 samples = {}".format(np.sum(
        ev_samples[1, :]) / 1000))

    fig, axs = plt.subplots(2, 2)

    for dim in range(2):

        print("Program 2: Marginal Expectation for dim {} for Graph Based with 1000 samples = {}".format(dim, np.sum(
            gb_samples[dim, :]) / 1000))

        print("Program 2: Marginal Expectation for dim {} for Eval Based with 1000 samples = {}".format(dim, np.sum(
            ev_samples[dim, :]) / 1000))

        axs[0, dim].hist(gb_samples[dim, :], bins=20)
        axs[0, dim].set_title('1000 Graph Based Samples Dim {}'.format(dim))
        axs[0, dim].set_xlabel('Sample Value Bins')
        axs[0, dim].set_ylabel('Frequency')

        axs[1, dim].hist(ev_samples[dim, :], bins=20)
        axs[1, dim].set_title('1000 Eval Based Samples Dim {}'.format(dim))
        axs[1, dim].set_xlabel('Sample Value Bins')
        axs[1, dim].set_ylabel('Frequency')

    plt.show()

    ############################################################################
    # Program 3
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW2/programs/3.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/3.daphne'])

    gb_samples = np.zeros((17, 1000))
    ev_samples = np.zeros((17, 1000))

    for j in range(1000):
        gb_samples[:, j] = sample_from_joint(graph).numpy()
        ev_samples[:, j] = evaluate_program(ast).numpy()

    fig0, axs0 = plt.subplots(5, 4)
    fig1, axs1 = plt.subplots(5, 4)
    fig0.tight_layout(pad=0.5)
    fig1.tight_layout(pad=0.5)

    for dim in range(17):

        print("Program 3: Marginal Expectation for dim {} for Graph Based with 1000 samples = {}".format(dim, np.sum(
            gb_samples[dim, :]) / 1000))

        print("Program 3: Marginal Expectation for dim {} for Eval Based with 1000 samples = {}".format(dim, np.sum(
            ev_samples[dim, :]) / 1000))

        axs0[int(dim / 4), int(dim % 4)].hist(gb_samples[dim, :], bins=20)
        axs0[int(dim / 4), int(dim % 4)].set_title('1000 Graph Based Samples Dim {}'.format(dim), fontsize=10)
        axs0[int(dim / 4), int(dim % 4)].set_xlabel('Sample Value Bins')
        axs0[int(dim / 4), int(dim % 4)].set_ylabel('Frequency')

        axs1[int(dim / 4), int(dim % 4)].hist(ev_samples[dim, :], bins=20)
        axs1[int(dim / 4), int(dim % 4)].set_title('1000 Eval Based Samples Dim {}'.format(dim), fontsize=10)
        axs1[int(dim / 4), int(dim % 4)].set_xlabel('Sample Value Bins')
        axs1[int(dim / 4), int(dim % 4)].set_ylabel('Frequency')

    fig0.delaxes(axs0[4, 1])
    fig1.delaxes(axs1[4, 1])
    fig0.delaxes(axs0[4, 2])
    fig1.delaxes(axs1[4, 2])
    fig0.delaxes(axs0[4, 3])
    fig1.delaxes(axs1[4, 3])

    plt.show()

    ############################################################################
    # Program 4
    graph = daphne(['graph', '-i', '../cpsc532w_hw/HW2/programs/4.daphne'])
    ast = daphne(['desugar', '-i', '../cpsc532w_hw/HW2/programs/4.daphne'])

    gb_w0 = np.zeros((10, 1000))
    gb_b0 = np.zeros((10, 1000))
    gb_w1 = np.zeros((10, 10, 1000))
    gb_b1 = np.zeros((10, 1000))

    ev_w0 = np.zeros((10, 1000))
    ev_b0 = np.zeros((10, 1000))
    ev_w1 = np.zeros((10, 10, 1000))
    ev_b1 = np.zeros((10, 1000))

    for j in range(1000):
        ev_w0[:, j] = evaluate_program(ast)[0].numpy().squeeze()
        ev_b0[:, j] = evaluate_program(ast)[1].numpy().squeeze()
        ev_w1[:, :, j] = evaluate_program(ast)[2].numpy()
        ev_b1[:, j] = evaluate_program(ast)[3].numpy().squeeze()

        gb_w0[:, j] = sample_from_joint(graph)[0].numpy().squeeze()
        gb_b0[:, j] = sample_from_joint(graph)[1].numpy().squeeze()
        gb_w1[:, :, j] = sample_from_joint(graph)[2].numpy()
        gb_b1[:, j] = sample_from_joint(graph)[3].numpy().squeeze()

    # Plot W1 stats
    ev_w1_mean = np.mean(ev_w1, axis=2)
    gb_w1_mean = np.mean(gb_w1, axis=2)

    ev_w1_std = np.std(ev_w1, axis=2)
    gb_w1_std = np.std(gb_w1, axis=2)

    ax0 = plt.axes()
    sns.heatmap(gb_w1_mean, ax=ax0, annot=True)
    ax0.set_title('W1 Mean for Graph Based Sampling')

    ax1 = plt.axes()
    sns.heatmap(gb_w1_std, ax=ax1, annot=True)
    ax1.set_title('W1 STD for Graph Based Sampling')

    ax2 = plt.axes()
    sns.heatmap(ev_w1_mean, ax=ax2, annot=True)
    ax2.set_title('W1 Mean for Eval Based Sampling')

    ax3 = plt.axes()
    sns.heatmap(ev_w1_std, ax=ax3, annot=True)
    ax3.set_title('W1 STD for Eval Based Sampling')

    # Plot the rest
    fig0, axs0 = plt.subplots(2, 5)
    fig1, axs1 = plt.subplots(2, 5)
    fig2, axs2 = plt.subplots(2, 5)
    fig3, axs3 = plt.subplots(2, 5)
    fig4, axs4 = plt.subplots(2, 5)
    fig5, axs5 = plt.subplots(2, 5)
    fig0.tight_layout(pad=0.5)
    fig1.tight_layout(pad=0.5)
    fig2.tight_layout(pad=0.5)
    fig3.tight_layout(pad=0.5)
    fig4.tight_layout(pad=0.5)
    fig5.tight_layout(pad=0.5)

    for dim in range(10):

        print("Program 4: Marginal Expectation for dim {} of w0 for Graph Based with 1000 samples = {}".format(dim, np.sum(
            gb_w0[dim, :]) / 1000))

        print("Program 4: Marginal Expectation for dim {} of b0 for Graph Based with 1000 samples = {}".format(dim, np.sum(
            gb_b0[dim, :]) / 1000))

        print("Program 4: Marginal Expectation for dim {} of b1 for Graph Based with 1000 samples = {}".format(dim, np.sum(
            gb_b1[dim, :]) / 1000))

        print("Program 4: Marginal Expectation for dim {} of w0 for Eval Based with 1000 samples = {}".format(dim, np.sum(
            ev_w0[dim, :]) / 1000))

        print("Program 4: Marginal Expectation for dim {} of b0 for Eval Based with 1000 samples = {}".format(dim, np.sum(
            ev_b0[dim, :]) / 1000))

        print("Program 4: Marginal Expectation for dim {} of b1 for Eval Based with 1000 samples = {}".format(dim, np.sum(
            ev_b1[dim, :]) / 1000))

        axs0[int(dim / 5), int(dim % 5)].hist(gb_w0[dim, :], bins=20)
        axs0[int(dim / 5), int(dim % 5)].set_title('1000 Graph Based Samples Dim {} of W0'.format(dim), fontsize=10)
        axs0[int(dim / 5), int(dim % 5)].set_xlabel('Sample Value Bins')
        axs0[int(dim / 5), int(dim % 5)].set_ylabel('Frequency')
        axs1[int(dim / 5), int(dim % 5)].hist(ev_w0[dim, :], bins=20)
        axs1[int(dim / 5), int(dim % 5)].set_title('1000 Eval Based Samples Dim {} of W0'.format(dim), fontsize=10)
        axs1[int(dim / 5), int(dim % 5)].set_xlabel('Sample Value Bins')
        axs1[int(dim / 5), int(dim % 5)].set_ylabel('Frequency')
        axs2[int(dim / 5), int(dim % 5)].hist(gb_b0[dim, :], bins=20)
        axs2[int(dim / 5), int(dim % 5)].set_title('1000 Graph Based Samples Dim {} of b0'.format(dim), fontsize=10)
        axs2[int(dim / 5), int(dim % 5)].set_xlabel('Sample Value Bins')
        axs2[int(dim / 5), int(dim % 5)].set_ylabel('Frequency')
        axs3[int(dim / 5), int(dim % 5)].hist(ev_b0[dim, :], bins=20)
        axs3[int(dim / 5), int(dim % 5)].set_title('1000 Eval Based Samples Dim {} of b0'.format(dim), fontsize=10)
        axs3[int(dim / 5), int(dim % 5)].set_xlabel('Sample Value Bins')
        axs3[int(dim / 5), int(dim % 5)].set_ylabel('Frequency')
        axs4[int(dim / 5), int(dim % 5)].hist(gb_b1[dim, :], bins=20)
        axs4[int(dim / 5), int(dim % 5)].set_title('1000 Graph Based Samples Dim {} of b1'.format(dim), fontsize=10)
        axs4[int(dim / 5), int(dim % 5)].set_xlabel('Sample Value Bins')
        axs4[int(dim / 5), int(dim % 5)].set_ylabel('Frequency')
        axs5[int(dim / 5), int(dim % 5)].hist(ev_b1[dim, :], bins=20)
        axs5[int(dim / 5), int(dim % 5)].set_title('1000 Eval Based Samples Dim {} of b1'.format(dim), fontsize=10)
        axs5[int(dim / 5), int(dim % 5)].set_xlabel('Sample Value Bins')
        axs5[int(dim / 5), int(dim % 5)].set_ylabel('Frequency')

    plt.show()