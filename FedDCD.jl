module FedDCD

using LinearAlgebra
using Printf
using SparseArrays
using Random
using StatsBase
using Distributed

export FedProxClient, FedDCDClient
export FedProxServer, FedDCDServer
export softmax, neg_log_loss
export split_data, splitDataByRow, labelTransform
export read_libsvm
export sgd!
export fedAvgAndProx, fedDCD


include("src/utils.jl")
include("src/LogReg/logReg.jl")
include("src/Client/client.jl")
include("src/Server/server.jl")
include("src/oracle.jl")
include("src/training.jl")


end # module