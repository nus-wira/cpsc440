using Printf
using Statistics
using Random
include("misc.jl")
include("clustering2Dplot.jl")

mutable struct PartitionModel
	predict # Function for clustering new points
	y # Cluster assignments
	W # Prototype points
end

function manhattanDistance(X, W)
	(n, d) = size(X)
	(t, d2) = size(W)
	@assert(d==d2)
	D = zeros(n, t)
	for i in 1:n
		for j in 1:t
			D[i,j] = sum(abs.(X[i,:]-W[j,:]))
		end
	end
	return D
end

function kMediansError(X, y, W)
	(n, d) = size(X)
	total = 0
	for i in 1:n
		total += sum(abs.(X[i,:] - W[y[i],:]))
	end
	return total
end

function kMedians(X,k;doPlot=false)
# K-medians clustering

(n,d) = size(X)
err = Inf

# Choos random points to initialize medians
W = zeros(k,d)
perm = randperm(n)
for c = 1:k
	W[c,:] = X[perm[c],:]
end

# Initialize cluster assignment vector
y = zeros(Int64, n)
changes = n
while changes != 0

	# Compute Manhattan distance between each point and each median
	D = manhattanDistance(X,W)

	# Degenerate clusters will distance NaN, change to Inf
	# (since Julia thinks NaN is smaller than all other numbers)
	D[findall(isnan.(D))] .= Inf

	# Assign each data point to closest median 
	# (track number of changes labels)
	changes = 0
	for i in 1:n
		(~,y_new) = findmin(D[i,:])
		changes += (y_new != y[i])
		y[i] = y_new
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end

	# Find median of each cluster
	for c in 1:k
		W[c,:] = median(X[y.==c,:],dims=1)
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end

	err = kMediansError(X, y, W)
	@printf("Running k-medians, kMediansError = %d\n",err)
end

function predict(Xhat)
	(t,d) = size(Xhat)

	D = distancesSquared(Xhat,W)
	D[findall(isnan.(D))] .= Inf

	yhat = zeros(Int64,t)
	for i in 1:t
		(~,yhat[i]) = findmin(D[i,:])
	end
	return yhat
end

return PartitionModel(predict,y,W), err
end
