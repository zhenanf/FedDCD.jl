mutable struct FedProxServer{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Vector{Int64}, T5<:Vector{FedProxClient}, T6<:Matrix{Float64}} <:AbstractServer
    Xtest::T3                        # testing data
    Ytest::T4                        # testing label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    participation_rate::T2           # participation rate
    clients::T5                      # set of clients
    W::T6                           # global models
    τ::T1                           # number of particiating clients
    selectedIndices::T4             # selected clients                    
    function FedProxServer(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, config::Dict{String, Real})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        participation_rate = config["participation_rate"]
        τ = floor(Int64, num_clients * participation_rate)
        _, d = size(Xtest)
        clients = Vector{FedProxClient}(undef, num_clients)
        W = zeros(Float64, d, num_classes)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Vector{Int64}, Vector{FedProxClient}, Matrix{Float64}}(Xtest, Ytest, num_classes, num_clients, participation_rate, clients, W, τ)
    end
end

# Select clients
function select!(
    server::FedProxServer
    )
    server.selectedIndices = randperm(server.num_clients)[1:server.τ]
    return nothing
end

# Send global model
function sendModel!(
    server::FedProxServer
)
    # Only send model to selected clients
    for idx in server.selectedIndices
        server.clients[idx].W = copy(server.W)
    end
    return nothing
end

# Send global model to all clients
function sendModelToAllClients!(
    server::FedProxServer
)
    # Only send model to selected clients
    for client in server.clients
        client.W = copy(server.W)
    end
    return nothing
end

# Aggregation
function aggregate!(
    server::FedProxServer
)
    # Take the average of the selected clients' model
    fill!(server.W, 0.0)
    for i = 1:server.τ
        idx = server.selectedIndices[i]
        server.W += server.clients[idx].W
    end
    server.W ./= server.τ
    return nothing
end

# Calculate objective value
function getObjValue(
    server::FedProxServer
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

########################################################################################################
mutable struct FedProxServerNN{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Flux.OneHotArray, T5<:Vector{FedProxClientNN}, T6<:Flux.Chain, T7<:Vector{Int64}} <:AbstractServer
    Xtest::T3                        # testing data
    Ytest::T4                        # testing label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    participation_rate::T2           # participation rate
    clients::T5                      # set of clients
    W::T6                           # global models
    τ::T1                           # number of particiating clients
    selectedIndices::T7             # selected clients                    
    function FedProxServerNN(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, dim::Int64, config::Dict{String, Real})
        num_classes = config["num_classes"]
        Ytest = Flux.onehotbatch(Ytest, 1:num_classes)
        num_clients = config["num_clients"]
        participation_rate = config["participation_rate"]
        W = Chain( Dense(780, dim, relu, bias=false), Dense(dim, 10, bias=false), NNlib.softmax)
        τ = floor(Int64, num_clients * participation_rate)
        clients = Vector{FedProxClientNN}(undef, num_clients)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Flux.OneHotArray, Vector{FedProxClientNN}, Flux.Chain, Vector{Int64}}(Xtest, Ytest, num_classes, num_clients, participation_rate, clients, W, τ)
    end
end

# Select clients
function select!(
    server::FedProxServerNN
    )
    server.selectedIndices = randperm(server.num_clients)[1:server.τ]
    return nothing
end

# Send global model, now exactly the same as `synchronize!`.
function sendModel!(
    server::FedProxServerNN
)
    # Only send model to selected clients
    for i in server.selectedIndices
        c = server.clients[i]
        for j = 1:length(params(server.W))
            params(c.W)[j] .= params(server.W)[j]
        end
    end
    return nothing
end

# Send global model to all clients
function sendModelToAllClients!(
    server::FedProxServerNN
)
    # Only send model to selected clients
    for c in server.clients
        for j = 1:length(params(server.W))
            params(c.W)[j] .= params(server.W)[j]
        end
    end
    return nothing
end

# Aggregation
function aggregate!(
    server::FedProxServerNN
)
    # Take the average of the selected clients' model
    l = length(params(server.W))
    for j = 1:l
        fill!(params(server.W)[j], 0.0)
    end
    for i = 1:server.τ
        idx = server.selectedIndices[i]
        c = server.clients[idx]
        for j = 1:l
            params(server.W)[j] .+= (1/server.τ) * params(c.W)[j]
        end
    end
    return nothing
end

# synchronization
function synchronize!(
    server::FedProxServerNN
)
    for i in server.selectedIndices
        c = server.clients[i]
        for j = 1:length(params(server.W))
            params(c.W)[j] .= params(server.W)[j]
        end
    end
    return nothing
end

# Calculate objective value
function getObjValue(
    server::FedProxServerNN
)
    objValue = 0.0
    totalNumData = 0
    for i = 1:server.num_clients
        c = server.clients[i]
        n = size(c.Xtrain, 1)
        totalNumData += n
        objValue += n * obj( c.XtrainT, c.Ytrain, server.W, c.λ)
    end
    return objValue / totalNumData
end