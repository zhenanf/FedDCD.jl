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

export FedProxClient, ScaffoldClient, FedDCDClient, AccFedDCDClient, getObjValue
export FedProxServer, ScaffoldServer, FedDCDServer, AccFedDCDServer, getObjValue
export softmax, neg_log_loss
export split_data, splitDataByRow, labelTransform, writeToFile, splitDataByClass
export read_libsvm
export fedAvgAndProx, Scaffold, fedDCD, accfedDCD
export obj, accuracy, getStochasticGrad, getGradient, Hv, ComputeNewtonDirection, ComputeNewtonDirection2, SoftmaxNewtonMethod, lineSearch, lineSearch2
export sgd!, newton!


include("src/utils.jl")
include("src/LogReg/logReg.jl")
include("src/oracle.jl")
include("src/Client/client.jl")
include("src/Server/server.jl")
include("src/training.jl")



end # module