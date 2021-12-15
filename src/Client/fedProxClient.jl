mutable struct FedProxClient{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Vector{Float64}, T5<:Vector{Int64}, T6<:Function} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    XtrainT::T3                             # Row copy
    Ytrain::T5                              # training label
    w::T4                                   # (model) primal variable
    lr::T2                                  # learning rate
    λ::T2                              # L2 regularization parameter
    μ::T2                                   # mu for proximal operation
    numLocalEpochs::T1                       # number of local steps
    function FedProxClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String,Real})
        num_classes = config["num_classes"]
        lambda = config["lambda"]
        learning_rate = config["learning_rate"]
        numLocalEpochs = config["numLocalEpochs"]
        μ = config["mu"]
        d = size(Xtrain, 2)
        w = zeros(Float64, d)
        XtrainT = copy(Xtrain')
        # y = zeros(Float64, num_classes, d)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Vector{Float64}, Vector{Int64}, Function}(id, Xtrain, XtrainT, Ytrain, w, learning_rate, λ, μ, numLocalEpochs)
    end
end