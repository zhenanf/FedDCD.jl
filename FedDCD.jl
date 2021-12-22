module FedDCD

using LinearAlgebra
using Printf
using SparseArrays
using Random
using StatsBase
using Distributed
using LinearMaps
using IterativeSolvers
using DataFrames
using CSV
using Flux
using Zygote

export FedProxClient, ScaffoldClient, FedDCDClient, AccFedDCDClient, getObjValue
export FedProxServer, ScaffoldServer, FedDCDServer, AccFedDCDServer, getObjValue
export FedDCDClientNN, FedDCDServerNN
export softmax, neg_log_loss
export split_data, splitDataByRow, labelTransform, writeToFile, splitDataByClass
export read_libsvm
export fedAvgAndProx, Scaffold, fedDCD, accfedDCD
export obj, accuracy
export getStochasticGrad, getGradient
export Hv, ComputeNewtonDirection, ComputeNewtonDirection2, SoftmaxNewtonMethod, lineSearch, lineSearch2
export dot_product
export sgd!, newton!, adam!


include("src/utils.jl")
include("src/LogReg/logReg.jl")
include("src/oracle.jl")
include("src/Client/client.jl")
include("src/Server/server.jl")
include("src/training.jl")



end # module