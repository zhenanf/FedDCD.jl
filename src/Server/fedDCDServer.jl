mutable struct FedDCDServer{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Vector{Int64}, T5<:Vector{FedDCDClient}, T6<:Matrix{Float64}} <:AbstractServer
    Xtest::T3                       # testing data
    Ytest::T4                       # testing label
    num_classes::T1                 # number of classes
    num_clients::T1                 # number of clients
    participation_rate::T2          # participation rate
    clients::T5                     # set of clients
    W::T6                           # aggregation of uploaded model updates
    τ::T1                           # number of particiating clients
    selectedIndices::T4             # selected clients
    η::T2                           # learning rate                    
    function FedDCDServer(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, config::Dict{String, Real})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        participation_rate = config["participation_rate"]
        η = config["learning_rate"]
        τ = floor(Int64, num_clients * participation_rate)
        _, d = size(Xtest)
        clients = Vector{FedDCDClient}(undef, num_clients)
        W = zeros(Float64, d, num_classes)
        selectedIndices = zeros(Int64, τ)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Vector{Int64}, Vector{FedDCDClient}, Matrix{Float64}}(Xtest, Ytest, num_classes, num_clients, participation_rate, clients, W, τ, selectedIndices, η)
    end
end

# Select clients
function select!(
    server::FedDCDServer
    )
    server.selectedIndices .= randperm(server.num_clients)[1:server.τ]
    return nothing
end

# Send global model
function sendModel!(
    server::FedDCDServer
)
    # Only send model to selected clients
    for i = 1:server.τ
        idx = server.selectedIndices[i]
        # server.clients[idx].W .-= copy(server.W)
        server.clients[idx].y .-= (server.η * server.clients[idx].λ) * (server.clients[idx].W - server.W)
    end
    return nothing
end

# Send global model to all clients
function sendModelToAllClients!(
    server::FedDCDServer
)
    # Only send model to selected clients
    for client in server.clients
        client.W = copy(server.W)
    end
    return nothing
end

# Aggregation
function aggregate!(
    server::FedDCDServer
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


########################################################################################################
mutable struct AccFedDCDServer{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Vector{Int64}, T5<:Vector{AccFedDCDClient}, T6<:Matrix{Float64}} <:AbstractServer
    Xtest::T3                       # testing data
    Ytest::T4                       # testing label
    num_classes::T1                 # number of classes
    num_clients::T1                 # number of clients
    participation_rate::T2          # participation rate
    clients::T5                     # set of clients
    W::T6                           # aggregation of uploaded model updates
    τ::T1                           # number of particiating clients
    selectedIndices::T4             # selected clients
    η::T2                           # learning rate                    
    function AccFedDCDServer(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, config::Dict{String, Real})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        participation_rate = config["participation_rate"]
        η = config["learning_rate"]
        τ = floor(Int64, num_clients * participation_rate)
        _, d = size(Xtest)
        clients = Vector{AccFedDCDClient}(undef, num_clients)
        W = zeros(Float64, d, num_classes)
        selectedIndices = zeros(Int64, τ)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Vector{Int64}, Vector{AccFedDCDClient}, Matrix{Float64}}(Xtest, Ytest, num_classes, num_clients, participation_rate, clients, W, τ, selectedIndices, η)
    end
end

# Select clients
function select!(server::AccFedDCDServer)
    server.selectedIndices .= randperm(server.num_clients)[1:server.τ]
    return nothing
end

# Send global model
# function sendModel!(server::AccFedDCDServer, round::Int64)
#     # Only send model to selected clients
#     for i = 1:server.τ
#         idx = server.selectedIndices[i]
#         if round == 1
#             server.clients[idx].y .-= (server.η * server.clients[idx].λ) * (server.clients[idx].W - server.W)
#         else
#             a = server.clients[idx].a; b = server.clients[idx].b; r = server.clients[idx].r
#             θ = a*r / (a^2 + b)
#             server.clients[idx].z .-= (server.η * server.clients[idx].λ * θ) * (server.clients[idx].W - server.W)
#         end
#     end
#     return nothing
# end

function sendModel!(server::AccFedDCDServer, round::Int64)
    # Only send model to selected clients, but all clients need to update 
    for i = 1:server.num_clients
        client = server.clients[i]
        if i in server.selectedIndices   # Can use store `selectedIndices` as Dict to improve efficency.
            if round == 1
                client.y .-= (server.η * client.λ) * (client.W - server.W)
            else
                a = client.a; b = client.b; r = client.r
                θ = a*r / (a^2 + b)
                client.z .-= (server.η * client.λ * θ) * (client.W - server.W)
            end
        else
            if round == 1
                client.y = copy(client.v)
            else
                client.z = copy(client.u)
            end
        end
    end
    return nothing
end

# Aggregation
function aggregate!(server::AccFedDCDServer)
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
    server::Union{FedDCDServer, AccFedDCDServer}
)
    objValue = 0.0
    totalNumData = 0
    for i = 1:server.num_clients
        n = size(server.clients[i].Xtrain, 1)
        totalNumData += n
        objValue += n * obj( 
            server.clients[i].Xtrain,
            server.clients[i].Ytrain,
            server.W,
            server.clients[i].λ
         )
        # objValue += n * getObjValue(server.clients[i])
    end
    return objValue / totalNumData
end