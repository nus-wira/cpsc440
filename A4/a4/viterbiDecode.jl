function viterbiDecode(p1,pt,d)
  k = length(p1)
  M = zeros(d, k)
  B = zeros(Int, d, k)
  x = zeros(Int, d)

  # Maximizing
  M[1,:] = p1
  for i in 2:d
    # each xi
    for j in 1:k
      # element wise
      p = pt[:, j] .* M[i-1,:]
      B[i, j] = argmax(p)
      M[i, j] = p[B[i,j]]
    end
  end

  # Backtracking
  x[d] = B[d, argmax(M[d, :])]
  for i in d-1:-1:1
    x[i] = B[x[i+1]]
  end

  return x
end