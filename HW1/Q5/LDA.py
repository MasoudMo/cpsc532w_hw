from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from joint_log_lik import joint_log_lik
from sample_topic_assignment import sample_topic_assignment
import time

bagofwords = loadmat('bagofwords_nips.mat')
WS = bagofwords['WS'][0] - 1  # go to 0 indexed
DS = bagofwords['DS'][0] - 1

WO = loadmat('words_nips.mat')['WO'][:, 0]
titles = loadmat('titles_nips.mat')['titles'][:, 0]

# This script outlines how you might create a MCMC sampler for the LDA model

alphabet_size = WO.size

document_assignment = DS
words = WS

# subset data, EDIT THIS PART ONCE YOU ARE CONFIDENT THE MODEL IS WORKING
# # PROPERLY IN ORDER TO USE THE ENTIRE DATA SET
# words = words[document_assignment < 100]
# document_assignment = document_assignment[document_assignment < 100]

n_docs = document_assignment.max() + 1

# number of topics
n_topics = 20

# initial topic assigments
topic_assignment = np.random.randint(n_topics, size=document_assignment.size)

# within document count of topics
doc_counts = np.zeros((n_docs, n_topics))

for d in range(n_docs):
    # histogram counts the number of occurences in a certain defined bin
    doc_counts[d] = \
    np.histogram(topic_assignment[document_assignment == d], bins=n_topics, range=(-0.5, n_topics - 0.5))[0]

# doc_N: array of size n_docs count of total words in each document, minus 1
doc_N = doc_counts.sum(axis=1) - 1

# within topic count of words
topic_counts = np.zeros((n_topics, alphabet_size))

for k in range(n_topics):
    w_k = words[topic_assignment == k]

    topic_counts[k] = np.histogram(w_k, bins=alphabet_size, range=(-0.5, alphabet_size - 0.5))[0]

# topic_N: array of size n_topics count of total words assigned to each topic
topic_N = topic_counts.sum(axis=1)

# prior parameters, alpha parameterizes the dirichlet to regularize the
# document specific distributions over topics and gamma parameterizes the
# dirichlet to regularize the topic specific distributions over words.
# These parameters are both scalars and really we use alpha * ones() to
# parameterize each dirichlet distribution. Iters will set the number of
# times your sampler will iterate.
alpha = 0.1 # As suggested in Wikipedia
gamma = 0.001 # As suggested in Wikipedia
iters = 100

jll = []
max_jll = -999999999999999
chosen_topic_counts = topic_counts
chosen_doc_counts = doc_counts
for i in range(iters):
    t_start = time.time()

    current_jll = joint_log_lik(doc_counts, topic_counts, alpha, gamma)
    jll.append(current_jll)

    # Store the sample with highest likelihood
    if current_jll > max_jll:
        max_jll = current_jll
        chosen_doc_counts = doc_counts
        chosen_topic_counts = topic_counts

    prm = np.random.permutation(words.shape[0])

    words = words[prm]
    document_assignment = document_assignment[prm]
    topic_assignment = topic_assignment[prm]

    topic_assignment, topic_counts, doc_counts, topic_N = sample_topic_assignment(
        topic_assignment,
        topic_counts,
        doc_counts,
        topic_N,
        doc_N,
        alpha,
        gamma,
        words,
        document_assignment)

    print('It took {} seconds for iteration {}. \n'.format((time.time() - t_start), i))

current_jll = joint_log_lik(doc_counts, topic_counts, alpha, gamma)
jll.append(current_jll)
if current_jll > max_jll:
    max_jll = current_jll
    chosen_doc_counts = doc_counts
    chosen_topic_counts = topic_counts

plt.plot(jll)
plt.show()

### find the 10 most probable words of the 20 topics:
fstr = ''
for k in range(n_topics):
    # Using the code excerpt found at:
    # https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
    frequent_words = (-chosen_topic_counts[k]).argsort()[:10]

    for wrd in frequent_words:
        fstr += str(WO[wrd][0]) + ', '

    fstr += '\n'

with open('most_probable_words_per_topic', 'w') as f:
    f.write(fstr)

# most similar documents to document 0 by cosine similarity over topic distribution:
# normalize topics per document and dot product:
number_of_docs = doc_counts.shape[0]
sim_vec = np.zeros(number_of_docs-1)
norm_doc_counts = chosen_doc_counts / np.sum(chosen_doc_counts, axis=1)[:, np.newaxis]
first_doc = norm_doc_counts[0]

fstr = ''
for doc in range(1, number_of_docs):
    sim_vec[doc-1] = np.dot(first_doc, norm_doc_counts[doc])

# Using the code excerpt found at:
# https://www.kite.com/python/answers/how-to-find-the-n-maximum-indices-of-a-numpy-array-in-python
most_sim_docs = (-sim_vec).argsort()[:10]

for doc in most_sim_docs:
    fstr += str(titles[doc][0]) + ', '

fstr += '\n'

with open('most_similar_titles_to_0', 'w') as f:
    f.write(fstr)
