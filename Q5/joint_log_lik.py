import numpy as np
from scipy.special import loggamma


def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model

    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics (sigma_{d,k})
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words (epsilon_{k,r})
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        ll: the joint log likelihood of the model
    """

    # We would like to compute log(p(w, z | \alpha, \gamma))
    # We have already derived a formula for p(w, z | \alpha, \gamma)

    ll = 0.0

    number_of_tops = topic_counts.shape[0]
    number_of_docs = doc_counts.shape[0]
    number_of_wrds = topic_counts.shape[1]

    # First term in the likelihood
    for d in range(number_of_docs):

        # Numerator of likelihood term
        for k in range(number_of_tops):
            ll += loggamma(doc_counts[d, k] + alpha)

        # Denominator of likelihood term
        ll -= loggamma(np.sum(doc_counts[d, :] + alpha))

    # Second term in the likelihood
    for k in range(number_of_tops):

        # Numerator of likelihood term
        for i in range(number_of_wrds):
            ll += loggamma(topic_counts[k, i] + gamma)

        # Denominator of likelihood term
        ll -= loggamma(np.sum(topic_counts[k, :] + gamma))

    return ll
