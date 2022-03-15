# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= 0
ytest[ytest.==2] .= 0
(n,d) = size(X)

# Choose network structure and randomly initialize weights
using Flux
global W1 = randn(28,d+1)
global W2 = randn(28,29)
v = randn(29)

global vt = reshape(v,1,29)
model = Chain(x -> W1*[x; 1], z -> relu.(z), x1 -> W2*[x1; 1], z -> relu.(z),a -> vt*[a; 1],  z -> sigmoid.(z))
loss(x,y) = (1/2)*(model(x)[1]-y)^2

# Train with stochastic gradient
maxIter = 200000
stepSize = 1e-2
for t in 1:maxIter
	i = rand(1:n)
	g_layer = gradient(params([W1,W2,vt])) do
		return loss(X[i,:],y[i])
	end

	global W1 -= stepSize * g_layer[W1]
	global W2 -= stepSize * g_layer[W2]
	global vt -= stepSize * g_layer[vt]

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		yhat = [model(x)[1] for x in eachrow(Xtest)] .>= 0.5
		err = sum(yhat .!= ytest)/size(Xtest,1)
		f = sum([loss(X[i,:],y[i]) for i in 1:n])

		@printf("Training iteration = %d, error rate = %.2f\n",t-1,err)
		@printf("Training error = %.2f\n", f)
	end
end
