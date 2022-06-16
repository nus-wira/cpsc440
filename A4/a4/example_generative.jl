using JLD, Printf, Statistics

# Load X and y variable
data = load("gaussNoise.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a KNN classifier
# k = 1
# include("knn.jl")
# model = knn(X,y,k)

# Fit a GDA classifier
# include("gda.jl")
# model = gda(X,y)

# Fit a TDA classifier
include("tda.jl")
model = tda(X,y)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
# @printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)
# @printf("Train Error with GDA: %.3f\n",trainError)
@printf("Train Error with TDA: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
# @printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)
# @printf("Test Error with GDA: %.3f\n",testError)
@printf("Test Error with TDA: %.3f\n",testError)
