include("studentT.jl")

function tda_predict(Xhat,X,y)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = maximum(y)

  subModel = Array{DensityModel}(undef,k)
  theta = zeros(k)

  for c in 1:k
    idx = findall(==(c), y)
    theta[c] = length(idx) / n
    Xc = X[idx,:]
    subModel[c] = studentT(Xc)
  end

  # PDFs where i,j corresponds to xi,class
  p = zeros(t,k)
  for c in 1:k
    p[:,c] = subModel[c].pdf(Xhat)
  end
  p .*= theta'

  idx = argmax(p,dims=2)
  yhat = zeros(t)
  for i in 1:t
    yhat[i] = idx[i][2]
  end
  return yhat

end

function tda(X,y)
  predict(Xhat) = tda_predict(Xhat,X,y)
  return GenericModel(predict)
end