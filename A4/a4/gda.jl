using Statistics, LinearAlgebra

function logprob(xi,theta,mu,Sigma)
  tot = log(theta) - 0.5 * logdet(Sigma) 
  tot += - 0.5 * (xi-mu)'*inv(Sigma)*(xi-mu)
  return tot
end

function gda_predict(Xhat,X,y)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = maximum(y)
  theta = zeros(k)
  mu = zeros(k,d)
  Sigma = zeros(k,d,d)
  for c in 1:k
    idx = findall(==(c), y)
    theta[c] = length(idx) / n
    Xc = X[idx,:]
    mu[c,:] = mean(Xc, dims=1)
    Sigma[c,:,:] = cor(Xc)
  end
  yhat = zeros(t)
  for i in 1:t
    bestlogprob = -Inf
    for c in 1:k
      clogprob = logprob(Xhat[i,:],theta[c],mu[c,:],Sigma[c,:,:])
      if clogprob > bestlogprob
        yhat[i] = c
        bestlogprob = clogprob
      end
    end
  end
  return yhat
end

function gda(X,y)
  # Implementation of GDA
  predict(Xhat) = gda_predict(Xhat,X,y)
  return GenericModel(predict)
end
