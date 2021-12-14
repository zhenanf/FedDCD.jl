module FedDCD

using LinearAlgebra
using Printf
using SparseArrays
using Random
using StatsBase
using Distributed

export Client
export Server
export softmax, neg_log_loss
export split_data
export read_libsvm


include("src/utils.jl")
include("src/Client/client.jl")
# include("src/Server/server.jl")
include("src/oracle.jl")
# include("src/training.jl")


end # module