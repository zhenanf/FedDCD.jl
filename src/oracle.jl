########################################################################
# Gradient oracles
########################################################################
using Printf

# run SVRG to obtain an exact/ineact dual gradient
function svrg(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, w::Matrix{Float64}, y::Matrix{Float64})
    1
end

# run Newton's method to obtain an exact/ineact dual gradient
function newton!(X::SparseMatrixCSC{Float64, Int64}, Xt::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, λ::Float64, W::Matrix{Float64}, y::Matrix{Float64}, η::Float64; T::Int64 = 50, tol::Float64=1e-6)
    for t = 1:T
        g = getGradient(X, Xt, Y, W, λ) - y
        gnorm = norm(g)
        # @printf("   gnorm: %4.4e\n", gnorm)
        if gnorm < tol
            break
        end
        t1 = time()
        D = ComputeNewtonDirection2( X, Xt, Y, W, λ, g)
        t2 = time()
        # @printf("Time spent on computing Newton direction: %4.4f\n", t2 - t1)
        η = lineSearch2(X, Y, y, D, W, g, λ)
        # η = 1.0
        W .-= η*D
        if t == T
            @warn("Reached maximum iteration in Newton's method.")
        end
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

# run adam to obtain an exact/ineact dual gradient
function adam!(Xt::SparseMatrixCSC{Float64, Int64}, Y::Flux.OneHotArray, W::Flux.Chain, y::Zygote.Params, λ::Float64; num_epoches::Int64=20)
    # data
    data = Flux.Data.DataLoader((Xt, Y), batchsize=128, shuffle=true)
    # loss
    sqnorm(w) = sum(abs2, w)
    loss(x, l) = Flux.crossentropy(W(x), l) + (λ/2) * sum(sqnorm, params(W)) - dot(params(W), y)
    # optimizer 
    opt = ADAMW(0.001, (0.89, 0.995), 1e-8)
    # train
    for t = 1:num_epoches
        Flux.train!(loss, params(W), data, opt)
        # @printf "epoch: %d, obj: %.2f\n" t loss(Xt, Y)
    end
end