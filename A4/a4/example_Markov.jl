# Load X and y variable
using JLD

# Load initial probabilities and transition probabilities of Markov chain
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])

# Set 'k' as number of states
k = length(p1)

# Confirm that initial probabilities sum up to 1
@show sum(p1)

# Confirm that transition probabilities sum up to 1, starting from each state
@show sum(pt,dims=2)

# Ancestral Sampling
include("sampleAncestral.jl")
@show sampleAncestral(p1,pt,50)

@show monteCarlo(p1,pt,50)

# CK equations
include("marginalCK.jl")
@show marginalCK(p1,pt,200)

# Viterbi Decoding
include("viterbiDecode.jl")
@show viterbiDecode(p1,pt,50)
@show viterbiDecode(p1,pt,100)

p13 = [0,0,1,0,0,0,0]

@show marginalCK(p13,pt,50)

# Monte Carlo Rejection
include("monteCarloRejection.jl")
@show monteCarloRejection(p1, pt)

# Forward Condition
include("forwardCondition.jl")
@show forwardCondition(p1,pt,5)