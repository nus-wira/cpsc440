# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit Naive Bayes model
X[X.>0.5] .= 2
X[X.<0.5] .= 1
X = Int64.(X)
Xtest[Xtest.>0.5] .= 2
Xtest[Xtest.<0.5] .= 1
Xtest = Int64.(Xtest)
# include("naiveNaiveBayes.jl")
# model = naiveNaiveBayes(X,y)
# include("naiveBayes.jl")
# model = naiveBayes(X,y)
include("vqnb.jl")
include("p_xyz_images.jl")

errors = zeros(5)
for i in 5:5
    local model, p_xyz = VQNB(X, y, i)

    ## Compute error on test data
    local yhat = model.predict(Xtest)
    local err = sum(yhat .!= ytest)/size(Xtest,1)
    errors[i] = err
    @printf("Error rate = %.2f\n",err)
    p_xyz_images(p_xyz)
end

for i in 2:5
    @printf("k = %d : Error rate = %.2f\n", i, errors[i])
end
