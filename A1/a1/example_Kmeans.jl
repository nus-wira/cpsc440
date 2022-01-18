# Load data
using JLD
X = load("clusterData2.jld","X")

# K-means clustering
k = 4
include("kMeans.jl")
# include("kMedians.jl")
err = Inf
for i in 1:50
    cmodel, cerr = kMeans(X,k,doPlot=true)
    # cmodel, cerr = kMedians(X,k,doPlot=true)
    if cerr < err
        global model = cmodel
        global err = cerr
    end
end
println("Lowest error: ", err)

y = model.predict(X)

include("clustering2Dplot.jl")
clustering2Dplot(X,y,model.W)

