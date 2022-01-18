using LinearAlgebra

function kMeansError(X,y,W)

(n,d) = size(X)
total = 0

for i in 1:n
    for j in 1:d
        e = X[i,j] - W[y[i],j]
        total += e * e
    end
end

return total

end