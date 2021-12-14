
# using client: AbstractClient

mutable struct FedAvgClient{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}, T6<:Function} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    Ytrain::T5                              # training label
    w::T4                                   # (model) primal variable
    lr::T2                                  # learning rate
    numLocalEpochs::T1                       # number of local steps
    # oracle::T6                              # gradient oracle
    function FedAvgClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        learning_rate = config["learning_rate"]
        numLocalEpochs = config["numLocalEpochs"]
        d = size(Xtrain, 1)
        w = zeros(Float64, num_classes, d)
        # y = zeros(Float64, num_classes, d)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}, Function}(id, Xtrain, Ytrain, w, learning_rate, numLocalEpochs)
    end
end

# Model update on local device
function update!(
    client::FedAvgClient
)
    # Implement K epochs SGD
    numData = size(Xtrain, 1)
    perm = collect(1:numData)
    for epoch = 1:client.numLocalEpochs
        shuffle!(perm)
        for i in perm

        end
    end
end