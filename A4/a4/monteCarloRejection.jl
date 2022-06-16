include("sampleAncestral.jl")

function monteCarloRejection(p1,pt,d=10,xi=6)
  p = zeros(length(p1))
  for i in 1:10000
    x = sampleAncestral(p1,pt,d)
    if x[d] == xi
      p[x[5]] += 1
    end
  end

  acc = sum(p)
  p /= acc

  return (acc, p)
end