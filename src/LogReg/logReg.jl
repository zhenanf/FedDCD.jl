using LinearAlgebra
using SparseArrays

# Calculate the objective of logistic regression:
# \frac{1}{n} \sum_{i=1}^n \log( 1 + \exp( -w^T x_i y_i ) ) + \lambda/2 \|w\|^2.
function obj(
    X::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
    λ::Float64
)
    n, _ = size(X)
    XW = X*W
    objval = 0.0
    for i = 1:n
        prob = softmax(XW[i,:])
        objval += -log(prob[ y[i] ])
    end
    objval /= n
    objval += λ/2*norm(W)^2
    return objval
end


# Calculate the accuracy
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


# Calculate the stochastic gradient
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



# Calculate gradient
function getGradient(
    X::SparseMatrixCSC{Float64, Int64},
    Xt::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
    λ::Float64
)
    n, d = size(X)
    _, K = size(W)
    g = zeros(Float64, d, K)
    XW = X*W
    for i = 1:n
        s = softmax( XW[i,:] )
        s[ y[i] ] -= 1
        x = Xt[:,i]
        g += (x*s')
    end
    g ./= n
    g += λ.*W
    return g
end


# Hessian-vector product for softmax regression.
function Hv(
    X::SparseMatrixCSC{Float64, Int64},
    Xt::SparseMatrixCSC{Float64, Int64},
    W::Matrix{Float64},
    λ::Float64,
    V::Matrix{Float64}
)
    @printf("Using naive Hessian-vector product oracle\n")
    # Time complexity is O( nnz(X)*K^2 )
    n, d = size(X)
    _, K = size(W)
    XW = X*W
    XV = X*V
    P = zeros(Float64, n, K)
    ret = zeros(Float64, d, K)
    for i = 1:n
        P[i,:] = softmax( XW[i,:] )
    end
    # Loop over K, fill ret column by column
    for k = 1:K
        # Loop over the column of W
        for kk = 1:K
            # Calculate D*X*V[:,kk]
            if kk == k
                D = P[:,k] .* ( 1 .- P[:, kk] )
            else
                D = -P[:,k] .* P[:, kk]
            end
            DXVkk = D .* XV[:, kk]
            ret[:, k] += ( Xt * DXVkk )
        end
    end
    ret ./= n
    ret += λ.*V
    return ret
end

# Add a dense matrix with a sparse matrix in place, A = A + x * b', x sparse, b dense.
function FastHvMatrixUpdate!(
    A::Matrix{Float64},
    x::SparseVector{Float64,Int64},
    b::Vector{Float64}
)
    I, V = findnz(x)
    K = length(b)
    @inbounds for col = 1:K 
        @inbounds for idx = 1:length(I)
            row = I[idx]
            v = V[idx]
            A[row, col] += v*b[col]
        end
    end
    return nothing
end

# Fast Hessian-vector product for softmax classification
function FastHv(
    X::SparseMatrixCSC{Float64, Int64},
    Xt::SparseMatrixCSC{Float64, Int64},
    W::Matrix{Float64},
    λ::Float64,
    V::Matrix{Float64}
)
    # Time complexity is O( nnz(X)*K )
    n, d = size(X)
    _, K = size(W)
    XW = X*W
    XV = X*V
    P = zeros(Float64, n, K)
    ret = zeros(Float64, d, K)
    for i = 1:n
        P[i,:] = softmax( XW[i,:] )
    end
    @inbounds for i = 1:n 
        # Complexity of the loop: nnz(xi)*K
        # Calculate (x_i x_i^T V ( Λ - p_i p_i^T ) )
        xi = Xt[:, i]
        ppi = P[i, :]
        # First calculate x_i^T V
        xtV = XV[i, :]
        # # Calculate x_i^T V ( Λ - p_i p_i^T )
        xtVppt = xtV .* ppi
        xtVppt -= dot(xtV, ppi) .* ppi
        # # Calculate x_i x_i^T V ( Λ - p_i p_i^T )
        FastHvMatrixUpdate!( ret, xi, xtVppt)
        # ret += xi*xtVppt'
    end
    ret ./= n
    ret += λ.*V
    return ret
end

# Compute Newton direction. Use conjugate gradient to solve the linear system H D = g.
function ComputeNewtonDirection(
    X::SparseMatrixCSC{Float64, Int64},
    Xt::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
    λ::Float64,
    g::Matrix{Float64}
)
    maxIter = 100
    n, d = size(X)
    _, K = size(W)
    # Compute gradient norm.
    gnorm = norm(g)
    tol = 1e-4

    # Conjugate gradient iterations.
    D = zeros(Float64, d, K)
    r = g
    p = copy(r)
    for i = 1:maxIter
        rnorm = norm(r)
        # @printf("rnorm: %4.4e\n", rnorm)
        if rnorm < tol*gnorm
            break
        end
        Hp = Hv(X, Xt, W, λ, p)
        α = dot(r, r) / dot(p, Hp)
        D += α.*p
        rnew = r - α.*Hp
        β = norm(rnew)^2/norm(r)^2
        p = rnew + β.*p
        r = rnew
    end
    return D
end


# Compute Newton direction. Use lsmr to solve the linear system H D = g.
function ComputeNewtonDirection2(
    X::SparseMatrixCSC{Float64, Int64},
    Xt::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
    λ::Float64,
    g::Matrix{Float64}
)
    n, d = size(X)
    _, K = size(W)
    
    H = LinearMap(v->vec(FastHv(X, Xt, W, λ, reshape(v,d,K))), d*K, issymmetric=true, isposdef=true)
    D = cg(H, vec(g), abstol=1e-4, reltol=1e-4, maxiter=100)

    return reshape(D, d, K)
end


# Using Newton's method to solve the softmax classification problem.
function SoftmaxNewtonMethod(
    X::SparseMatrixCSC{Float64, Int64},
    Xt::SparseMatrixCSC{Float64, Int64},
    y::Vector{Int64},
    W::Matrix{Float64},
    λ::Float64,
    maxIter::Int64=20,
    tol::Float64=1e-4
)
    startTime = time()
    for iter = 1:maxIter
        objval = obj(X, y, W, λ)
        g = getGradient(X, Xt, y, W, λ)
        gnorm = norm(g)
        @printf("Iter %3d, obj: %4.5e, gnorm: %4.5e, time: %4.2f\n", iter, objval, gnorm, time()-startTime)
        if gnorm < tol
            break
        end
        # Compute Newton direction.
        D = ComputeNewtonDirection2( X, Xt, y, W, λ, g)
        # Use stepsize 1.
        W -=  D
    end
    return W
end