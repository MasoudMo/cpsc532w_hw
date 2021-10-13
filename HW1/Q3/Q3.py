import numpy as np
##Q3


##first define the probability distributions as defined in the excercise:

#define 0 as false, 1 as true
def p_C(c):
    p = np.array([0.5,0.5])
    
    return p[c]


def p_S_given_C(s,c):
    p = np.array([[0.5,0.9],[0.5,0.1]])
    return p[s,c]
    
def p_R_given_C(r,c):
    p = np.array([[0.8,0.2],[0.2,0.8]])
    return p[r,c]

def p_W_given_S_R(w,s,r):
    
    p = np.array([
            [[1.0,0.1],[0.1,0.01]],   #w = False  Had to fix this by changing 0.001 to 0.01
            [[0.0,0.9],[0.9,0.99]],   #w = True
            ])
    return p[w,s,r]


##1. enumeration and conditioning:
    
## compute joint:
p = np.zeros((2,2,2,2)) #c,s,r,w
for c in range(2):
    for s in range(2):
        for r in range(2):
            for w in range(2):
                p[c,s,r,w] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)*p_W_given_S_R(w,s,r)
                
## condition and marginalize:
# We are looking to compute p[C|W] = p[C,W]/p[W] first and then find p[C|W=T] and then get p[C=T|W=T]

# Let's compute p[C,W] first by marginalizing S and R out of the joint
p_joint_C_W = np.zeros((2, 2))
for s in range(2):
    for r in range(2):
        for w in range(2):
            for c in range(2):
                p_joint_C_W[c, w] += p[c, s, r, w]

# Let's compute p[W] now by marginalizing C and S and R out of the joint
p_W = np.zeros(2)
for c in range(2):
    for r in range(2):
        for s in range(2):
            for w in range(2):
                p_W[w] += p[c, s, r, w]

# Let's compute p[C|W=T]
p_C_given_W = np.zeros(2)
for c in range(2):
    p_C_given_W[c] = p_joint_C_W[c, 1] / p_W[1]

print('There is a {:.2f}% chance it is cloudy given the grass is wet'.format(p_C_given_W[1]*100))


##2. ancestral sampling and rejection:
num_samples = 10000
samples = np.zeros(num_samples)
rejections = 0
i = 0

# To perform conditional rejection sampling, if we desire to sample from p[x|y'], we first sample from from p[x, y]
# and only accept samples where y=y'


# Very inefficient rejection sampling function assuming that we can sample from a uniform distribution
def rejection_sampling_with_uniform(prob_to_sample):

    while 1:
        # sample from q (uniform)
        x = np.random.randint(0, 2)

        # sample uniformly from kq
        u = np.random.uniform()

        if u <= prob_to_sample[x]:
            return x


while i < num_samples:

    # Ancestral sampling from the joint (each term is sampled using rejection sampling)
    # P[C, S, R, W] = p[C]*p[S|C]*p[R|C]*p[W|S, R]

    # Sample from P[C] - We can just use uniform sampling in this case since P[C] = [0.5 0.5]
    c = np.random.randint(0, 2)

    # Sample from p[S|C=c] using rejection sampling
    s = rejection_sampling_with_uniform(p_S_given_C([0, 1], c))

    # Sample from p[R|C]
    r = rejection_sampling_with_uniform(p_R_given_C([0, 1], c))

    # Sample from p[W|S, R]
    w = rejection_sampling_with_uniform(p_W_given_S_R([0, 1], s, r))

    if w:
        samples[i] = c
        i = i+1
    else:
        rejections += 1


print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
print('{:.2f}% of the total samples were rejected'.format(100*rejections/(samples.shape[0]+rejections)))


#3: Gibbs
# we can use the joint above to condition on the variables, to create the needed
# conditional distributions:


#we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
#indexing is [c,s,r,w]
p_R_given_C_S_W = p/p.sum(axis=2, keepdims=True)
p_S_given_C_R_W = p/p.sum(axis=1, keepdims=True)


# but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
# but since p(W|S,R) does not depend on C, we can factor it out:
#p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

#first create p(R,S,C):
p_C_S_R = np.zeros((2,2,2)) #c,s,r
for c in range(2):
    for s in range(2):
        for r in range(2):
            p_C_S_R[c,s,r] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)
            
#then create the conditional distribution:
p_C_given_S_R = p_C_S_R[:,:,:]/p_C_S_R[:,:,:].sum(axis=(0),keepdims=True)

##gibbs sampling
num_samples = 10000
samples = np.zeros(num_samples)
state = np.zeros(4,dtype='int')
#c,s,r,w, set w = True

# Perform Gibbs sampling by sampling from the conditionals sequentially

# Sample x(0) from a uniform distribution
state[0] = np.random.randint(0, 2) # c
state[1] = np.random.randint(0, 2) # s
state[2] = np.random.randint(0, 2) # r
state[3] = 1 # w = True

for i in range(num_samples):

    # Sample from p[C|S,R]
    state[0] = rejection_sampling_with_uniform(p_C_given_S_R[:, state[1], state[2]])

    # Sample from p[S|C,R,W]
    state[1] = rejection_sampling_with_uniform(p_S_given_C_R_W[state[0], :, state[2], state[3]])

    # Sample from p[R|C,S,W]
    state[2] = rejection_sampling_with_uniform(p_R_given_C_S_W[state[0], state[1], :, state[3]])

    # Store the obtained c in the samples
    samples[i] = state[0]

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
