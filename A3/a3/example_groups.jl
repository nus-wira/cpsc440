using JLD
using SpecialFunctions
data = load("cancerData.jld")
n = data["n"]
nTest = data["nTest"]

# Pool data
# n = sum(n, dims=1)
# nTest = sum(nTest, dims=1)

# Compute number of groups (here, the number of training and testing groups is the same)
k = size(n,1)

# Function to compute NLL when we have a theta for each group
NLLs(theta,n) =
begin
    LL = 0
    for j in 1:k
        LL -= n[j,1]*log(theta[j]) + (n[j,2]-n[j,1])*log(1-theta[j])
    end
    return LL
end

# Show test NLL if all theta=0.5
theta = 0.5*ones(k)
@show NLLs(theta,nTest)

# Fit MLE for each training group
MLEs = n[:,1]./n[:,2]

# Show training and test NLL for MLE
@show NLLs(MLEs,nTest)

# Fit Laplace Smoothing
lap = (n[:,1].+1)./(n[:,2].+2)

# Show test NLL for Laplace Smoothing
@show NLLs(lap,nTest)

# Fit Posterior Predictive
a = 2
b = 2
postpre(a,b) = (n[:,1].+a)./(n[:,2].+(a+b))

# Show test NLL for posterior predictive
@show NLLs(postpre(2,2),nTest)

m = MLEs[1]
# k = 1000000000000000000000

logMargLik(a,b) = logbeta(a+n[1], b+n[2]) - logbeta(a,b)

objFn(a,b) = -logMargLik(a,b) + 0.99*log(a/(a+b)) - 8.9*log(1-a/(a+b)) + 2*log(1+a+b)

r = -Inf

# for i in 0.1:0.1:100
#     for j in 0.1:0.1:100
#         cr = objFn(i, j)
#         if cr < r
#             global r = cr
#             global a = i
#             global b = j
#             println("r= ", r)
#             println("a= ", a)
#             println("b= ", b)
#             println()
#         end
#     end
# end

# println("r= ", r)
# println("a= ", a)
# println("b= ", b)

objFnSep(a,b,n) = 
begin
    LL = 0
    for j in 1:k
        LL -= logbeta(a+n[j,1],b+n[j,2]-n[j,1]) - logbeta(a,b)
    end
    LL -= - 0.99*log(a/(a+b)) + 8.9*log(1-a/(a+b)) - 2*log(1+a+b)
end

a = 1
b = 731

probb = objFnSep(a, b, nTest)
@show objFnSep(a, b, nTest)

# for j in 1:1:1000
#     for i in 1:1000
#         cr = objFnSep(i,j)
#         if cr > r
#             continue
#         end
#         global r = cr
#         global a = i
#         global b = j
#         println("r= ", r)
#         println("a= ", a)
#         println("b= ", b)
#         println()
#     end
# end

# println("r= ", r)
# println("a= ", a)
# println("b= ", b)

println("postpred= ",postpre(a,b))

# probb=NLLs(postpre(a,b),nTest)
# @show NLLs(postpre(a,b),nTest)
# println(exp(-probb))

MAPs = (n[:,1].+(a+1))./(n[:,2].+(a+b+2))

@show NLLs(MAPs,nTest)