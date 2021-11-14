from evaluator import evaluate
import torch
import numpy as np
import json
import sys
from daphne import daphne
from copy import deepcopy


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done': True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res


def resample_particles(particles, log_weights):
    num_particles = len(particles)

    # Resample particles based on weights
    log_weights = torch.stack(log_weights)
    weights = np.exp(log_weights.detach().numpy())
    new_particle_idx = np.random.choice(range(num_particles), num_particles, p=weights/np.sum(weights))

    new_particles = list()
    for idx in new_particle_idx:
        new_particles.append(particles[idx])

    # Compute logZ
    logZ = np.log(np.sum(weights)/num_particles)

    return logZ, new_particles


def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                # Handle the observe case
                particles[i] = res
                sigma = res[2]

                # Calculate weight
                weights[i] = sigma['d'].log_prob(sigma['c'])

                # Make sure all threads converge at the same observe statement
                if i == 0:
                    alpha_cur = sigma['alpha']
                else:
                    if sigma['alpha'] != alpha_cur:
                        raise RuntimeError("Failed SMC, threads stopped at different observe statements")

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(1, 5):
        exp = daphne(['desugar-hoppl-cps', '-i', '../cpsc532w_hw/HW6/programs/{}.daphne'.format(i)])

        for n_particles in [1, 10, 100, 1000, 10000, 100000]:

            logZ, particles = SMC(n_particles, exp)

            print('logZ: ', logZ)

            values = torch.stack(particles)


