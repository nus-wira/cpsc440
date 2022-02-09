# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= 0
ytest[ytest.==2] .= 0
(n,d) = size(X)

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [28,28]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 200000
stepSize = 1e-2
for t in 1:maxIter
	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
	global w = w - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		# yhat = sign.(NeuralNet_predict(w,Xtest,nHidden))
		yhat = NeuralNet_predict(w,Xtest,nHidden) .>= 0.5
		err = sum(yhat .!= ytest)/size(Xtest,1)
		if err < 0.025
			global stepSize = 1e-2
		end
		@printf("Training iteration = %d, error rate = %.2f\n",t-1,err)
		@printf("Training error = %.2f\n", f)
	end
end
