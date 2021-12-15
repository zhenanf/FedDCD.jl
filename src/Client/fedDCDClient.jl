mutable struct FedDCDClient{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}, T6<:Function} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    XtrainT::T3                             # Row copy
    Ytrain::T5                              # training label
    W::T4                                   # (model) primal variable
    y::T4                                   # (model) dual variable
    η::T2                                   # learning rate
    λ::T2                                   # L2 regularization parameter
    oracle!::T6                              # gradient oracle
    function FedDCDClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String, Real}, oracle!::Function)
        numClasses = config["num_classes"]
        λ = config["lambda"]
        η = config["learning_rate"]
        d = size(Xtrain, 2)
        W = zeros(Float64, d, numClasses)
        y = zeros(Float64, d, numClasses)
        XtrainT = copy(Xtrain')
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}, Function}(id, Xtrain, XtrainT, Ytrain, W, y, η, λ, oracle!)
    end
end

# Model update on local device
function update!(
    client::FedDCDClient
)
    @printf("Client %d locally update\n", client.id)
    client.oracle!(client.XtrainT, client.Ytrain, client.λ, client.W, client.y, client.η)
end

# Get objective value
function getObjValue(
    client::FedDCDClient
)
    objValue = obj(client.Xtrain, client.Ytrain, client.W, client.λ)
    return objValue
end