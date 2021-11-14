from evaluator import evaluate
import torch
import numpy as np
import json
import sys
from daphne import daphne


def get_IS_sample(exp):
    # The output is the identity
    output = lambda x: x

    # Get the output of evaluate
    res = evaluate(exp, env=None)('addr_start', output)

    # Keep running until res is not a continuation anymore
    logW = 0.0
    while type(res) is tuple:
        cont, args, sigma = res
        if sigma['type'] == 'observe':
            logW += sigma['d'].log_prob(sigma['c'])
        res = cont(*args)

    return logW, res

if __name__ == '__main__':

    for i in range(1, 5):
        exp = daphne(['desugar-hoppl-cps', '-i', '../cpsc532w_hw/HW6/programs/{}.daphne'.format(i)])

        print('\n\n\nSample of posterior of program {}:'.format(i))
        log_weights = []
        values = []
        for i in range(10000):
            logW, sample = get_IS_sample(exp)
            log_weights.append(logW)
            values.append(sample)

        log_weights = torch.tensor(log_weights)

        values = torch.stack(values)
        values = values.reshape((values.shape[0],values.size().numel()//values.shape[0]))
        log_Z = torch.logsumexp(log_weights,0) - torch.log(torch.tensor(log_weights.shape[0],dtype=float))

        log_norm_weights = log_weights - log_Z
        weights = torch.exp(log_norm_weights).detach().numpy()
        weighted_samples = (torch.exp(log_norm_weights).reshape((-1,1))*values.float()).detach().numpy()
    
        print('covariance: ', np.cov(values.float().detach().numpy(), rowvar=False, aweights=weights))
        print('posterior mean:', weighted_samples.mean(axis=0))
