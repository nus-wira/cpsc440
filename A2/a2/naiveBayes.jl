include("misc.jl") # Includes GenericModel typedef

function naiveBayes(X,y)
    # Implementation of generative classifier,
    # where a product of Bernoullis is used for p(x,y)

    (n,d) = size(X)
    # Number of each class
    n1 = sum(y.==1)
    n2 = n - n1

    # Compute p(y = 1)
    p_y = sum(y.==1)/n

    # We will store p(x(i,j) = 1|y) in p_xy(j)
    p_xy = zeros(d, 2)
    # Indices of each class
    i1 = findall(==(1), y)
    i2 = findall(!=(1), y)

    for j in 1:d
        p_xy[j, 1] = sum(X[i1, j].==1)/n1
        p_xy[j, 2] = sum(X[i2, j].==1)/n2
    end

    function predict(Xhat)
        (t,d) = size(Xhat)
        yhat = zeros(t)

        for i in 1:t
            # p_yx = p_y*prod(p_x) for the appropriate x and y values
            p_yx = [p_y;1-p_y]
            for j in 1:d
                if Xhat[i,j] == 1
                    p_yx[1] *= p_xy[j,1]
                    p_yx[2] *= p_xy[j,2]
                else
                    p_yx[1] *= 1-p_xy[j,1]
                    p_yx[2] *= 1-p_xy[j,2]
                end
            end
            if p_yx[1] > p_yx[2]
                yhat[i] = 1
            else
                yhat[i] = 2
            end
        end
        return yhat
    end

    return GenericModel(predict)
end
