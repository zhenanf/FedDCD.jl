mutable struct ScaffoldServer{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Vector{Int64}, T5<:Vector{ScaffoldClient}, T6<:Matrix{Float64}} <:AbstractServer
    Xtest::T3                        # testing data
    Ytest::T4                        # testing label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    participation_rate::T2           # participation rate
    clients::T5                      # set of clients
    W::T6                           # global models
    C::T6                           # Control variate
    τ::T1                           # number of particiating clients
    lr::T2                          # global learning rate                 
    selectedIndices::T4             # selected clients   
    function ScaffoldServer(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, config::Dict{String, Real})
        num_classes = config["num_classes"] 
        num_clients = config["num_clients"]
        participation_rate = config["participation_rate"]
        lr = config["learning_rate"]
        τ = floor(Int64, num_clients * participation_rate)
        _, d = size(Xtest)
        clients = Vector{ScaffoldClient}(undef, num_clients)
        W = zeros(Float64, d, num_classes)
        C = zeros(Float64, d, num_classes)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Vector{Int64}, Vector{ScaffoldClient}, Matrix{Float64}}(Xtest, Ytest, num_classes, num_clients, participation_rate, clients, W, C, τ, lr)
    end
end

# Select clients
function select!(
    server::ScaffoldServer
    )
    server.selectedIndices = randperm(server.num_clients)[1:server.τ]
    return nothing
end

# Send global model
function sendModel!(
    server::ScaffoldServer
)
    # Only send model to selected clients
    for idx in server.selectedIndices
        server.clients[idx].W .= copy(server.W)
    end
    return nothing
end

# Send global model to all clients
function sendModelToAllClients!(
    server::ScaffoldServer
)
    # Only send model to selected clients
    for client in server.clients
        client.W .= copy(server.W)
    end
    return nothing
end

# Aggregation
function aggregate!(
    server::ScaffoldServer
)
    # Take the average of the selected clients' model
    d, K = size( server.W )
    ΔW = zeros(Float64, d, K)
    ΔC = zeros(Float64, d, K)
    for i = 1:server.τ
        idx = server.selectedIndices[i]
        ΔW += server.clients[idx].ΔW
        ΔC += server.clients[idx].ΔC
    end
    ΔW ./= server.τ
    ΔC ./= server.τ
    # Update
    server.W .+= server.lr*ΔW
    server.C .+= server.lr*ΔC
    return nothing
end

# Calculate objective value
function getObjValue(
    server::ScaffoldServer
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