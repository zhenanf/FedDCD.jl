########################################################################
# Gradient oracles
########################################################################

# run SVRG to obtain an exact/ineact dual gradient
function svrg(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, w::Matrix{Float64}, y::Matrix{Float64})
    1
end

# run Newton's method to obtain an exact/ineact dual gradient
function newton(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, w::Matrix{Float64}, y::Matrix{Float64})
    1
end

# run SGD to obtain an exact/ineact dual gradient
function sgd!(Xt::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, λ::Float64, W::Matrix{Float64}, y::Matrix{Float64}, η::Float64; T::Int64 = 20)
    # using lazy update to avoid dense update from the regularization
    d, n = size(Xt)
    _, k = size(W)
    hitTime = zeros(Int64, d, k)
    perm = collect(1:n)
    for t = 1:T
        fill!(hitTime, 0)
        shuffle!(perm)
        timeStep = 1
        for i in perm
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
        end
        # Lazy update for staled coordinates
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
    return nothing
end