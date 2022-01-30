include("misc.jl") # Includes GenericModel typedef
include("kMeans.jl")

function VQNB(X,y,k)
    # Implementation of generative classifier,
    # where a product of Bernoullis is used for p(x,y)
    # in clusters

    (n,d) = size(X)
    # Number of each class
    n1 = sum(y.==1)
    n2 = n - n1
    
    # Indices of each class
    i1 = findall(==(1), y)
    i2 = findall(!=(1), y)

    X1 = X[i1,:]
    X2 = X[i2,:]
    y1 = y[i1]
    y2 = y[i2]

    model1 = kMeans(X1,k)
    model2 = kMeans(X2,k)

    # Clusters for each class
    z1 = model1.predict(X1)
    z2 = model2.predict(X2)

    X12 = [X1; X2]
    y12 = [y1; y2]
    z12 = [z1; z2]

    # Hold num of examples of a cluster for each class
    # nz[c,b] is # yi = b and zi = c
    # nzd[j,b,c] is # yi = b and zi = c and xij = 1
    nz = zeros(k,2)
    nzd = zeros(d,2,k)
    for i in 1:n
        for z in 1:k
            if z12[i] != z
                continue
            end
            nz[z,y12[i]] += 1
            for j in 1:d
                nzd[j,y12[i],z] += X12[i,j] == 1
            end
        end
    end

    # Compute p(y = 1)
    p_y = sum(y.==1)/n

    # Compute p(z = c | yi = b)
    p_zy = nz ./ [n1 n2]

    # We will store p(x(i,j) = 1|y,z) in p_xyz(j)
    p_xyz = zeros(d, 2, k)

    for j in 1:d
        p_xyz[j,:,:] = nzd[j,:,:] ./ nz'
    end

    function predict(Xhat)
        (t,d) = size(Xhat)
        yhat = zeros(t)

        for i in 1:t
            # p_yx = p_y*prod(p_x) for the appropriate x and y values
            p_yx = [p_y;1-p_y]
            p_yzyxyz = [p_y;1-p_y]

            # Sum of p_zy * prod(p_xyz)
            p_zyxyz = zeros(2)
            for z in 1:k
                p_yxyz = p_zy[z,:]
                # Product of p_xyz
                for j in 1:d
                    if Xhat[i,j] == 1
                        p_yxyz[1] *= p_xyz[j,1,z]
                        p_yxyz[2] *= p_xyz[j,2,z]
                    else
                        p_yxyz[1] *= 1-p_xyz[j,1,z]
                        p_yxyz[2] *= 1-p_xyz[j,2,z]
                    end
                end
                p_zyxyz += p_yxyz
            end

            p_yzyxyz .*= p_zyxyz

            if p_yzyxyz[1] > p_yzyxyz[2]
                yhat[i] = 1
            else
                yhat[i] = 2
            end
        end
        return yhat
    end

    return GenericModel(predict)
end
