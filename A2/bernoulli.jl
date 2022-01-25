using Plots

function generate(t, theta = 0.12)
    return rand(t) .<= theta
end

function approx(t)
    gen = generate(t)
    n1 = sum(gen)
    n0 = t - n1
    return (n0 - 5*n1) / t
end

function plotApprox(fn)
    x = 1:100000
    y = fn.(x)
    plot(x, y)
    return x, y
end

# plotApprox(approx)

function approx10(t)
    v = zeros(t)

    # Get cummulative sums first
    for i in 1:t
        tot = 0
        n = 0
        while tot < 10
            tot += generate(1)[1]
            n += 1
        end
        v[i] = n
        if i > 1
            v[i] += v[i-1]
        end
    end

    # Then get running approximation
    for i in 1:t
        v[i] /= i
    end

    return v
end

function plotApprox10()
    x = 1:100000
    y = approx10(100000)
    plot(x, y)
    return y[100000]
end

print(plotApprox10())