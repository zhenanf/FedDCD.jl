########################################################################
# Gradient oracles
########################################################################
include("LogReg/logReg.jl")

# run SVRG to obtain an exact/ineact dual gradient
function svrg(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, w::Matrix{Float64}, y::Matrix{Float64})
    1
end

# run Newton's method to obtain an exact/ineact dual gradient
function newton!(X::SparseMatrixCSC{Float64, Int64}, Xt::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, λ::Float64, W::Matrix{Float64}, y::Matrix{Float64}, η::Float64; T::Int64 = 5, tol::Float64=1e-4)
    for t = 1:T
        g = getGradient(X, Xt, Y, W, λ) - y
        gnorm = norm(g)
        @printf("gnorm: %4.4e\n", gnorm)
        if gnorm < tol
            break
        end
        D = ComputeNewtonDirection( X, Xt, Y, W, λ, g)
        W .-=  D
    end
end

# run SGD to obtain an exact/ineact dual gradient
function sgd!(X::SparseMatrixCSC{Float64, Int64}, Xt::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, λ::Float64, W::Matrix{Float64}, y::Matrix{Float64}, η::Float64; T::Int64 = 20, is_lazy::Bool = true)
    n, d = size(X)
    _, k = size(W)
    if is_lazy
        hitTime = zeros(Int64, d, k)
    end
    perm = collect(1:n)
    for t = 1:T
        shuffle!(perm)
        if is_lazy
            fill!(hitTime, 0)
            timeStep = 1
        end  
        for i in perm
            if is_lazy
                g = getStochasticGrad(Xt, Y, W, i)
                I, J, V = findnz(g)
                for j = 1:length(I)
                    idx1 = I[j]
                    idx2 = J[j]
                    # Lazy update
                    W[idx1, idx2] *= (1 - η*λ)^(timeStep-hitTime[idx1, idx2])
                    W[idx1, idx2] += (timeStep-hitTime[idx1, idx2])*η*y[idx1, idx2]
                    W[idx1, idx2] -= η*V[j]
                    # Update hitTime to the current time
                    hitTime[idx1, idx2] = timeStep
                end
                timeStep += 1
            else
                g = getStochasticGrad(Xt, Y, W, i) + λ*W - y
                W .-= η*g
            end
        end
        # Lazy update for staled coordinates
        if is_lazy
            timeStep -= 1
            for i = 1:d
                for j = 1:k
                    if hitTime[i, j] < timeStep
                        W[i, j] *= (1 - η*λ)^(timeStep-hitTime[i, j])
                        W[i, j] += (timeStep-hitTime[i, j])*η*y[i, j]
                    end
                end
            end
        end
    end
    return nothing
end