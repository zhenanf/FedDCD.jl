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
    ΔC ./= server.num_clients
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


########################################################################################################
mutable struct ScaffoldServerNN{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Zygote.Params, T5<:Vector{ScaffoldClientNN}, T6<:Flux.Chain, T7<:Vector{Int64}, T8<:Flux.OneHotArray} <:AbstractServer
    Xtest::T3                        # testing data
    Ytest::T8                        # testing label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    participation_rate::T2           # participation rate
    clients::T5                      # set of clients
    W::T6                           # global models
    C::T4                           # Control variate
    τ::T1                           # number of particiating clients
    lr::T2                          # global learning rate                 
    selectedIndices::T7             # selected clients   
    function ScaffoldServerNN(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, dim::Int64, config::Dict{String, Real})
        num_classes = config["num_classes"]
        Ytest = Flux.onehotbatch(Ytest, 1:num_classes) 
        num_clients = config["num_clients"]
        participation_rate = config["participation_rate"]
        lr = config["learning_rate"]
        τ = floor(Int64, num_clients * participation_rate)
        # Set up clients and initialize models and control variates.
        clients = Vector{ScaffoldClient}(undef, num_clients)
        W = Chain( Dense(780, dim, relu, bias=false), Dense(dim, 10, bias=false), NNlib.softmax)
        C = deepcopy( params(W) )
        for j = 1:length(C)
            fill!( C[j], 0.0 )
        end
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Zygote.Params, Vector{ScaffoldClientNN}, Flux.Chain, Vector{Int64}, Flux.OneHotArray}(Xtest, Ytest, num_classes, num_clients, participation_rate, clients, W, C, τ, lr)
    end
end


# Select clients
function select!(
    server::ScaffoldServerNN
    )
    server.selectedIndices = randperm(server.num_clients)[1:server.τ]
    return nothing
end

# Send global model
function sendModel!(
    server::ScaffoldServerNN
)
    # Only send model to selected clients
    for i in server.selectedIndices
        c = server.clients[i]
        for j = 1:length(params(server.W))
            params(c.W)[j] .= params(server.W)[j]
            c.globalC[j] .= server.C[j]
        end
    end
    return nothing
end

# Send global model to all clients
function sendModelToAllClients!(
    server::ScaffoldServerNN
)
    # Only send model to selected clients
    for c in server.clients
        for j = 1:length(params(server.W))
            params(c.W)[j] .= params(server.W)[j]
            c.globalC[j] .= server.C[j]
        end
    end
    return nothing
end

# Aggregation
function aggregate!(
    server::ScaffoldServerNN
)
    # Take the average of the selected clients' model
    # @show( length(params(server.W)) )
    # @show( length(server.C) )
    l = length(params(server.W))
    # for j = 1:l
    #     fill!(params(server.W)[j], 0.0)
    #     fill!(server.C[j], 0.0)
    # end
    for i = 1:server.τ
        idx = server.selectedIndices[i]
        c = server.clients[idx]
        # @show( length(c.ΔW) )
        # @show( length(c.ΔC) )
        for j = 1:l 
            params(server.W)[j] .+= (server.lr/server.τ) * c.ΔW[j]
            server.C[j] .+= (server.lr/server.num_clients) * c.ΔC[j] 
        end
    end
    return nothing
end

# Calculate objective value
function getObjValue(
    server::ScaffoldServerNN
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