import numpy as np

def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.

    Args:
        topic_assignment: size n array of topic assignments (topic for each word)
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words (epsilon_{k,r})
        doc_counts: n_docs x n_topics array of counts per document of unique topics (sigma_{d,k})

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1

        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of words
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """

    number_of_tops = topic_counts.shape[0]
    alphabet_size = topic_counts.shape[1]

    topic_dist = np.zeros(number_of_tops)

    # Iterating over all words in each document
    for i, w in enumerate(words):
        # Subtract the word count associated with the z we are sampling
        doc_counts[document_assignment[i], topic_assignment[i]] -= 1
        topic_counts[topic_assignment[i], w] -= 1
        topic_N[topic_assignment[i]] -= 1

        # Assign a value to probability of each topic for z of this word
        for k in range(number_of_tops):
            topic_dist[k] = ((doc_counts[document_assignment[i], k] + alpha) /
                             (doc_N[document_assignment[i]] + number_of_tops*alpha)) * \
                            ((topic_counts[k, w] + gamma)/(topic_N[k] + alphabet_size*gamma))

        # Normalize the obtained distribution
        topic_dist /= np.sum(topic_dist)

        # Sample a new z from categorical distribution
        sampled_z = np.random.choice(range(topic_counts.shape[0]), p=topic_dist)

        # Update counts according to newly sampled z
        doc_counts[document_assignment[i], sampled_z] += 1
        topic_counts[sampled_z, w] += 1
        topic_N[sampled_z] += 1

        # Update assignment vectors
        topic_assignment[i] = sampled_z

    return topic_assignment, topic_counts, doc_counts, topic_N




