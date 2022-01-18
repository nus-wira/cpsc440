include("misc.jl")
include("findMin.jl")

function softmaxClassifier(X,y)
    (n,d) = size(X)
    k = maximum(y)
    W = ones(d, k)
    W = reshape(W, :, 1)

    funObj(W) = softmaxObj(W, X, y)
    W = findMin(funObj, W, derivativeCheck=true)
    W = reshape(W, d, k)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)
	return LinearModel(predict,W)
end

function softmaxObj(W, X, y)
    (n,d) = size(X)
    k = maximum(y)
    W = reshape(W, d, k)
    t = sum(-log.(softmaxProb(W, X, y)))
    g = softmaxGrad(W, X, y)
    g = reshape(g, :, 1)
    return t, g
end

function softmaxProb(W, X, y)
    n = size(X, 1)
    p = zeros(n, 1)
    for i in 1:n
        p[i] = softmaxProbSingle(W, X[i, :], y[i])
    end
    return p
end

function softmaxProbSingle(W, xi, yi)
    return exp(W[:,yi]'*xi) / sum(exp.(xi'*W))
end

function softmaxGrad(W, X, y)
    (n, d) = size(X)
    k = size(W, 2)
    g = zeros(d, k)
    for c in 1:k
        wc = zeros(d, 1)
        yc = ones(n,1) # Treat class 'c' as +1
        yc[y .!= c] .= 0 # Treat other classes as 0
        for i in 1:n
            wc += (softmaxProbSingle(W, X[i,:], c) - yc[i])*X[i,:]
        end
        g[:,c] = wc
    end
    
    return g
end