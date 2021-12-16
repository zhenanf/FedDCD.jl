mutable struct FedProxClient{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}, T6<:Function} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    XtrainT::T3                             # Row copy
    Ytrain::T5                              # training label
    W::T4                                   # (model) primal variable
    lr::T2                                  # learning rate
    λ::T2                              # L2 regularization parameter
    μ::T2                                   # mu for proximal operation
    numLocalEpochs::T1                       # number of local steps
    function FedProxClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String,Real})
        num_classes = config["num_classes"]
        λ = config["lambda"]
        learning_rate = config["learning_rate"]
        numLocalEpochs = config["numLocalEpochs"]
        μ = config["mu"]
        d = size(Xtrain, 2)
        W = zeros(Float64, d, num_classes)
        XtrainT = copy(Xtrain')
        # y = zeros(Float64, num_classes, d)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}, Function}(id, Xtrain, XtrainT, Ytrain, W, learning_rate, λ, μ, numLocalEpochs)
    end
end

# Model update on local device
function update!(
    client::FedProxClient
)
    @printf("Client %d running SGD\n", client.id)
    # Implement K epochs SGD, using lazy update to avoid dense update from the regularization.
    n, d = size(client.Xtrain)
    _, K = size(client.W)
    lr = client.lr
    Wold = copy(client.W)
    λ = client.λ
    μ = client.μ
    hitTime = zeros(Int64, d, K)
    perm = collect(1:n)
    for epoch = 1:client.numLocalEpochs
        fill!(hitTime, 0)
        shuffle!(perm)
        timeStep = 1
        for i in perm
            g = getStochasticGrad(client.XtrainT, client.Ytrain, client.W, i)
            I, J, V = findnz(g)
            for j = 1:length(I)
                idx1 = I[j]
                idx2 = J[j]
                # Lazy update
                delay = timeStep-hitTime[idx1, idx2]
                client.W[idx1, idx2] *= (1 - lr*(λ+μ))^delay
                # client.W[idx1, idx2] += lr*μ*delay*Wold[idx1, idx2]
                client.W[idx1, idx2] -= lr*V[j]
                # Update hitTime to the current time
                hitTime[idx1, idx2] = timeStep
            end
            timeStep += 1
        end
        # Lazy update for staled coordinates
        timeStep -= 1
        for j = 1:d
            for k = 1:K
                if hitTime[j, k] < timeStep
                    delay = timeStep-hitTime[j, k]
                    client.W[j, k] *= (1 - lr*λ)^delay
                    # client.W[j, k] += lr*μ*delay*Wold[j, k]
                end
            end
        end
    end
    return nothing
end


# Get objective value
function getObjValue(
    client::FedProxClient
)
    objValue = obj(client.Xtrain, client.Ytrain, client.W, client.λ)
    return objValue
end