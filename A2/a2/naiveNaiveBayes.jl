include("misc.jl") # Includes GenericModel typedef

function naiveNaiveBayes(X,y)
	# Implementation of generative classifier,
	# where a product of Bernoullis is used for p(x,y)

	(n,d) = size(X)

	# Compute p(y = 1)
	p_y = sum(y.==1)/n

	# We will store p(x(i,j) = 1) in p_x(j)
	p_x = zeros(d)
	for j in 1:d
		p_x[j] = sum(X[:,j].==1)/n
	end

	function predict(Xhat)
		(t,d) = size(Xhat)
		yhat = zeros(t)

		for i in 1:t
			# p_yx = p_y*prod(p_x) for the appropriate x and y values
			p_yx = [p_y;1-p_y]
			for j in 1:d
				if Xhat[i,j] == 1
					p_yx[1] *= p_x[j]
					p_yx[2] *= p_x[j]
				else
					p_yx[1] *= 1-p_x[j]
					p_yx[2] *= 1-p_x[j]
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
