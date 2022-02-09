using Printf
using Statistics

# Load X and y variable
using JLD
using PyPlot
pygui(true)
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
Xn = X .- mean(X)
Xn ./= std(X)
yn = y .- mean(y)
yn ./= std(y)
(n,d) = size(X)
println(d)
# Adding bias variable
X = [X ones(n)]
Xn = [Xn ones(n)]
d += 1

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [11]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)
# w = randn(nParams, 1) .- 0.5

# Train with stochastic gradient
maxIter = 10000
stepSize = 1e-2
for t in 1:maxIter

	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
	# (f,g) = NeuralNet_backprop(w,Xn[i,:],yn[i],nHidden)
	global w = w - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		@printf("Training iteration = %d\n",t-1)
		@printf("f = %f\n", f)
		figure(1)
		clf()
		xVals = -10:.05:10
		Xhat = zeros(length(xVals),1)
		Xhat[:] .= xVals
		# Bias variable
		# yhat = NeuralNet_predict(w,Xhat,nHidden)
		yhat = NeuralNet_predict(w,[Xhat ones(length(xVals))],nHidden)
		# Xhatn = Xhat .- mean(X)
		# Xhatn ./= std(X)
		# Xhatn = [Xhatn ones(length(xVals))]
		# yhatn = NeuralNet_predict(w,Xhatn,nHidden)
		# yhat = yhatn .* std(y)
		# yhat .+= mean(y)
		# plot(X,y,".")
		plot(X[:,1:end-1],y,".")
		plot(Xhat,yhat,"g-")
		sleep(.1)
	end
end
