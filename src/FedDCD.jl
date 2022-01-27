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
using Zygote
using Flux

export FedProxClient, ScaffoldClient, FedDCDClient, AccFedDCDClient, getObjValue
export FedProxServer, ScaffoldServer, FedDCDServer, AccFedDCDServer, getObjValue
export FedDCDClientNN, FedDCDServerNN, AccFedDCDClientNN, AccFedDCDServerNN
export FedProxClientNN, FedProxServerNN, ScaffoldClientNN, ScaffoldServerNN
export softmax, neg_log_loss
export split_data, splitDataByRow, labelTransform, writeToFile, splitDataByClass
export read_libsvm
export fedAvgAndProx, Scaffold, fedDCD, accfedDCD
export obj, accuracy
export getStochasticGrad, getGradient
export ComputeNewtonDirection, SoftmaxNewtonMethod, lineSearch, lineSearch2
export dot_product
export sgd!, newton!, adam!


include("./utils.jl")
include("./LogReg/logReg.jl")
include("./oracle.jl")
include("./Client/client.jl")
include("./Server/server.jl")
include("./training.jl")



end # module