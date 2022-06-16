
function marginalCK(p1,pt,d)
    p = p1
    for i in 2:d
        p = pt' * p
    end
    return p
end
