include("misc.jl")

function leastSquares(X,y)
	(n, d) = size(X)
	Z = [ones(n, 1) X]

	# Find regression weights minimizing squared error
	# w = (X'*X)\(X'*y)
	v = (Z'*Z)\(Z'*y)


	# Make linear prediction function
	# predict(Xtilde) = Xtilde*w
	predict(Xtilde) = [ones(size(Xtilde,1), 1) Xtilde]*v

	# Return model
	return LinearModel(predict,v)
end
