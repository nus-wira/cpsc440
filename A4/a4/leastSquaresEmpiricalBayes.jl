include("leastSquares.jl")

function leastSquaresEmpiricalBayes(X,y)
  (n,d) = size(X)
  bLL = -Inf
  blambda = 0
  bsigma = 0
  bp = 0
  
  for p in 1:10
    k = p + 1
    Z = polyBasis(X,p)
    lambda = 0.25
    for i in 1:10
      lambda *= 2
      sigma = 0.25
      for j in 1:10
        sigma *= 2
        Theta = (1/sigma^2)*Z'*Z + lambda * I
        w = (1/sigma^2)*inv(Theta)*Z'*y
        LL = (k/2) * log(lambda) - n * log(sigma) - 0.5 * logdet(Theta)
        LL += (-1/(2*sigma^2)) * norm(Z*w - y)^2
        LL += (-lambda/2) * norm(w)^2
        if LL > bLL
          bLL = LL
          blambda = lambda
          bsigma = sigma
          bp = k
          println(bLL, ",", bp, ",", blambda, ",", bsigma)
        end
      end
    end
  end

  return (bLL, bp, blambda, bsigma)
end