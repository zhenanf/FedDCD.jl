using LinearAlgebra
using SparseArrays

@doc raw"""
    Calculate the objective of logistic regression:
    \frac{1}{n} \sum_{i=1}^n \log( 1 + \exp( -w^T x_i y_i ) ) + \lambda/2 \|w\|^2.
    """
function obj(
    X::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    w::Vector{Float64},
    lambda::Float64
)
    n, _ = size(X)
    Xw = X*w
    yXw = y.*Xw
    tmp =  1 .+ exp.(-yXw)
    objval = sum( log.( tmp ) ) / n + lambda/2*norm(w)^2
    return objval
end

"""
    Calculate the accuracy
"""
function accuracy(
    X::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    w::Vector{Float64},
)
    n = size(X, 1)
    yhat = X*w
    return sum( yhat.*y .> 0 )/n
end

"""
    Calculate the stochastic gradient
"""
function getStochasticGrad(
    Xt::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    w::Vector{Float64},
    idx::Int64
)
    x = Xt[:,idx]
    xw = dot( x, w )
    yxw = y[idx]*xw
    p =  1 / (1 + exp(-yxw))
    g = y[idx]*(p-1).*x
    return g
end