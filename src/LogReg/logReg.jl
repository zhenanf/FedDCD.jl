using LinearAlgebra
using SparseArrays

@doc raw"""
    Calculate the objective of logistic regression:
    \frac{1}{n} \sum_{i=1}^n \log( 1 + \exp( -w^T x_i y_i ) ) + \lambda/2 \|w\|^2.
    """
function obj(
    X::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
    lambda::Float64
)
    n, _ = size(X)
    XW = X*W
    objval = 0.0
    for i = 1:n
        prob = softmax(XW[i,:])
        objval += -log(prob[ y[i] ])
    end
    objval /= n
    objval += lambda/2*norm(W)^2
    return objval
end

"""
    Calculate the accuracy
"""
function accuracy(
    X::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
)
    n = size(X, 1)
    XW = X*W
    ret = 0
    for i = 1:n
        prob = softmax( XW[i,:] )
        if argmax(prob) == y[i]
            ret += 1
        end
    end
    return ret/n
end

"""
    Calculate the stochastic gradient
"""
function getStochasticGrad(
    Xt::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
    idx::Int64
)
    x = Xt[:,idx]
    Wtx = W'*x
    s = softmax( Wtx )
    s[ y[idx] ] -= 1
    g = x * s'
    return g
end

