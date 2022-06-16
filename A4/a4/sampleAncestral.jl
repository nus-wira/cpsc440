include("misc.jl")

function sampleAncestral(p1,pt,d)
    x = zeros(Int,d)
    x[1] = sampleDiscrete(p1)
    for i in 2:d
        x[i] = sampleDiscrete(pt[x[i-1],:])
    end
    return x
end

function monteCarlo(p1,pt,d)
    p = zeros(length(p1))
    for i in 1:10000
        x = sampleAncestral(p1,pt,d)
        p[x[d]] += 1
    end
    return p / sum(p)
end