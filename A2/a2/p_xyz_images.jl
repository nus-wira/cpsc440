using PyPlot
pygui(true)

function p_xyz_images(p_xyz)
    (d,c,k) = size(p_xyz)

    for z in 1:k
        for i in 1:c
            img = p_xyz[:,i,z]
            img = reshape(img, 28, 28)'
            imsave("p_xyz_$(z)_$(i).png", img, cmap="gray")
            # imshow(img, "gray")
        end
    end
end