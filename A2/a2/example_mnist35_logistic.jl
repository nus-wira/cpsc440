# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])


if true
## Fit logistic regression model
include("logreg.jl")
model = logReg12(X,y)

## Compute error on test data
yhat = model.predict(Xtest)
err = sum(yhat .!= ytest)/size(Xtest,1)
@printf("Error rate = %.2f\n",err)
end

# Fit Naive Bayes model
X[X.>0.5] .= 2
X[X.<0.5] .= 1
X = Int64.(X)
Xtest[Xtest.>0.5] .= 2
Xtest[Xtest.<0.5] .= 1
Xtest = Int64.(Xtest)
include("naiveBayes.jl")
include("vqnb.jl")
#model = naiveBayes(X,y)
#model = naiveBayesLaplace(X,y,1)
#model = CANB(X,y,1)
model = VQNB(X,y,10)

@printf("Predicting...\n")

## Compute error on train data
#yhat = model.predict(X)
#err = sum(yhat .!= y)/size(X,1)
#@printf("Error rate = %.2f\n",err)

## Compute error on test data
yhat = model.predict(Xtest)
err = sum(yhat .!= ytest)/size(Xtest,1)
@printf("Error rate = %.2f\n",err)
