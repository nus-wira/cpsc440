# We use nHidden as a vector, containing the number of hidden units in each layer
# Definitely not the most efficient implementation!

dropout = 0

# Function that returns total number of parameters
function NeuralNet_nParams(d,nHidden)

	# Connections from inputs to first hidden layer
	nParams = d*nHidden[1]

	# Connections between hidden layers
	for h in 2:length(nHidden)
		# nParams += nHidden[h-1]*nHidden[h]
		nParams += (1+nHidden[h-1])*nHidden[h]
	end

	# Connections from last hidden layer to output
	# nParams += nHidden[end]
	nParams += 1 + nHidden[end]

end

# Compute squared error and gradient
# for a single training example (x,y)
# (x is assumed to be a column-vector)
function NeuralNet_backprop(bigW,x,y,nHidden)
	d = length(x)
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	# - This is not a really elegant way to do things
	# if you want to be really efficient, but the course
	# it is nice abraction
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		# Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*(nHidden[layer-1]+1)],nHidden[layer],nHidden[layer-1]+1)
		# ind += nHidden[layer]*nHidden[layer-1]
		ind += nHidden[layer]*(nHidden[layer-1]+1)
	end
	v = bigW[ind+1:end]

	#### Define activation function and its derivative
	# h(z) = tanh.(z)
	# dh(z) = (sech.(z)).^2
	function relu(x)
		return max(x, 0)
	end
	h(z) = relu.(z)
	dh(z) = z .>= 0
	# Sigmoid
	# h(z) = 1 ./ (1 .+ exp.(-z))
	# dh(z) = h(z) .* (1 .- h(z))

	#### Forward propagation
	z = Array{Any}(undef,nLayers)
	z[1] = W1*x
	# Dropout
	z[1] .*= rand(nHidden[1]) .>= dropout
	z[1] ./= (1-dropout)
	for layer in 2:nLayers
		# z[layer] = Wm[layer-1]*h(z[layer-1])
		z[layer] = Wm[layer-1]*[h(z[layer-1]); 1]
		# Dropout
		z[layer] .*= rand(nHidden[layer]) .>= dropout
		z[layer] ./= (1-dropout)
	end
	# yhat = v'*h(z[end])
	yhat = v'*[h(z[end]); 1]
	yhat = 1/(1+exp(-yhat))

	r = yhat-y
	# Regularize
	f = (1/2)r^2
	# f = -y*log(yhat) - (1-y)*log(1-yhat)
	# f += 0.005 * sum(bigW.^2)

	#### Backpropagation (the below could be replaced by AD)
	dr = r
	err = dr

	# Output weights
	# Gout = err*h(z[end])
	Gout = err*h([z[end];1])

	Gm = Array{Any}(undef,nLayers-1)
	if nLayers > 1
		# Last Layer of Hidden Weights
		# backprop = err*(dh(z[end]).*v)
		# Gm[end] = backprop*h(z[end-1])'
		backprop = err*(dh(z[end]).*v[1:end-1])
		Gm[end] = backprop*h([z[end-1];1])'

		# Other Hidden Layers
		for layer in nLayers-2:-1:1
			# backprop = (Wm[layer+1]'*backprop).*dh(z[layer+1])
			# Gm[layer] = backprop*h(z[layer])'
			backprop = (Wm[layer+1][:,1:end-1]'*backprop).*dh(z[layer+1])
			Gm[layer] = backprop*h([z[layer];1])'
		end

		# Input Weights
		# backprop = (Wm[1]'*backprop).*dh(z[1])
		backprop = (Wm[1][:,1:end-1]'*backprop).*dh(z[1])
		G1 = backprop*x'
	else
		# Input weights
		# G1 = err*(dh(z[1]).*v)*x'
		G1 = err*(dh(z[1]).*v[1:end-1])*x'
	end

	#### Put gradients into vector
	g = zeros(size(bigW))
	g[1:nHidden[1]*d] = G1
	ind = nHidden[1]*d
	for layer in 2:nLayers
		# g[ind+1:ind+nHidden[layer]*nHidden[layer-1]] = Gm[layer-1]
		g[ind+1:ind+nHidden[layer]*(nHidden[layer-1]+1)] = Gm[layer-1]
		ind += nHidden[layer]*(nHidden[layer-1]+1)
	end
	g[ind+1:end] = Gout
	# g *= 1.01

	return (f,g)
end

# Computes predictions for a set of examples X
function NeuralNet_predict(bigW,Xhat,nHidden)
	(t,d) = size(Xhat)
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		# Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*(nHidden[layer-1]+1)],nHidden[layer],nHidden[layer-1]+1)
		ind += nHidden[layer]*(nHidden[layer-1]+1)
	end
	v = bigW[ind+1:end]

	#### Define activation function and its derivative
	# h(z) = tanh.(z)
	# dh(z) = (sech.(z)).^2
	function relu(x)
		return max(x, 0)
	end
	h(z) = relu.(z)
	dh(z) = z .>= 0
	# Sigmoid
	# h(z) = 1 ./ (1 .+ exp.(-z))
	# dh(z) = h(z) .* (1 .- h(z))

	#### Forward propagation on each example to make predictions
	yhat = zeros(t,1)
	for i in 1:t
		# Forward propagation
		z = Array{Any}(undef,1nLayers)
		z[1] = W1*Xhat[i,:]
		z[1] *= 1-dropout
		for layer in 2:nLayers
			# z[layer] = Wm[layer-1]*h(z[layer-1])
			z[layer] = Wm[layer-1]*h([z[layer-1];1])
			z[layer] .*= 1-dropout
		end
		# yhat[i] = v'*h(z[end])
		yhat[i] = v'*h([z[end];1])
		yhat[i] = 1/(1+exp(-yhat[i]))
	end
	return yhat
end

