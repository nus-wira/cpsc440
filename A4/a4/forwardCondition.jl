include("marginalCK.jl")

function forwardCondition(p1,pt,c,d=10,xi=6)
  k = length(p1)

  pc = marginalCK(p1,pt,c)
  pd = zeros(k)
  pd[xi] = 1
  
  for i in d:-1:c+1
    pd = pt * pd
  end

  condp = pc .* pd

  return condp / sum(condp)

end