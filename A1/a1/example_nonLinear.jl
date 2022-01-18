# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Fit least squares model
# include("leastSquares.jl")
# model = leastSquares(X,y)

using Random
m = div(n,2)
Xtr = X[1:m,:]
Xv = X[m+1:n,:]
y = reshape(y, (length(y), 1))
ytr = y[1:m,:]
yv = y[m+1:n,:]
err = Inf

# Fit least squares model
include("leastSquaresRBFL2.jl")
include("misc.jl")
# model = leastSquaresRBFL2(X,y,1,1)

function getModel(Xtr, ytr, Xv, yv)
    s = 0
    l = 0
    err = Inf
    for i in 1:20
        sigma = 0.5 * i
        for j in 1:20
            lambda = 0.1 * j

            cmodel = leastSquaresRBFL2(Xtr,ytr,sigma,lambda)
            
            local yhat = cmodel.predict(Xv)
            yhat = reshape(yhat, (length(yhat), 1))

            cerr = distancesSquared(yhat, yv)[1,1]
            # println(cerr)
            if cerr < err
                global model = cmodel
                global err = cerr
                global s = sigma
                global l = lambda
                
                println("err: ", err)
                println("sigma: ", s)
                println("lambda: ", l)
            end
        end
    end
    println("found sigma & lambda: ", s, ",", l)
    println("error:", err)
    return model
end

model = getModel(Xtr, ytr, Xv, yv)

# Plot model
using PyPlot
pygui(true)
figure()
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((0,2.5))
