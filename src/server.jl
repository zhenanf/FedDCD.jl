########################################################################
# Server
########################################################################

mutable struct Server{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Vector{Int64}, T5<:Vector{Client}, T6<:Vector{Matrix{Float64}}} 
    Xtest::T3                        # training data
    Ytest::T4                        # training label
    num_classes::T1                  # number of classes
    num_clients::T1                  # number of clients
    learning_rate::T2                # learning rate
    participation_rate::T2           # participation rate
    clients::T5                      # set of clients
    ws::T6                           # uploaded models                    
    function Server(Xtest::SparseMatrixCSC{Float64, Int64}, Ytest::Vector{Int64}, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        num_clients = config["num_clients"]
        learning_rate = config["learning_rate"]
        participation_rate = config["participation_rate"]
        clients = Vector{Client}(undef, num_clients)
        ws = Vector{Matrix{Float64}}(undef, round(participation_rate*num_clients))
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Vector{Int64}, Vector{Client}, Vector{Matrix{Float64}}}(Xtest, Ytest, num_classes, num_clients, learning_rate, participation_rate, clients, ws)
    end
end