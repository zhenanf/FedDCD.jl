mutable struct FedAvgServer{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Vector{Int64}, T5<:Vector{FedAvgClient}, T6<:Vector{Float64}} <:AbstractServer
    Xtest::T3                        # testing data
    Ytest::T4                        # testing label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    participation_rate::T2           # participation rate
    clients::T5                      # set of clients
    w::T6                           # global models
    τ::T1                           # number of particiating clients
    selectedIndices::T4             # selected clients                    
    function FedAvgServer(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, config::Dict{String, Real})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        participation_rate = config["participation_rate"]
        τ = floor(Int64, num_clients * participation_rate)
        _, d = size(Xtest)
        clients = Vector{FedAvgClient}(undef, num_clients)
        w = zeros(Float64, d)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Vector{Int64}, Vector{FedAvgClient}, Vector{Float64}}(Xtest, Ytest, num_classes, num_clients, participation_rate, clients, w,τ)
    end
end

# Select clients
function select!(
    server::FedAvgServer
    )
    server.selectedIndices = randperm(server.num_clients)[1:server.τ]
    return nothing
end

# Send global model
function sendModel!(
    server::FedAvgServer
)
    # Only send model to selected clients
    for idx = 1:server.num_clients
        server.clients[idx].w = copy(server.w)
    end
    return nothing
end

# Aggregation
function aggregate!(
    server::FedAvgServer
)
    # Take the average of the selected clients' model
    fill!(server.w, 0.0)
    for i = 1:server.τ
        idx = server.selectedIndices[i]
        server.w += server.clients[idx].w
    end
    server.w ./= server.τ
    return nothing
end

# Calculate objective value
function getObjValue(
    server::FedAvgServer
)
    objValue = 0.0
    totalNumData = 0
    for i = 1:server.num_clients
        n = size(server.clients[i].Xtrain, 1)
        totalNumData += n
        objValue += n * getObjValue(server.clients[i])
    end
    return objValue / totalNumData
end