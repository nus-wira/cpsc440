using LinearAlgebra
include("misc.jl")

function leastSquaresRBFL2(X, y, sigma, lambda)

    Z = exp.(distancesSquared(X, X) / (2*sigma*sigma))
    v = (Z'*Z + lambda*I)\(Z'*y)

    predict(Xt) = exp.(distancesSquared(Xt, X) / (2*sigma*sigma)) * v

    return LinearModel(predict, v)
end