########################################################################
# Client
########################################################################\
module Client

using LinearAlgebra
using SparseArrays
using Random

export AbstractClient

# Abstract class for client
abstract type AbstractClient end

# Client upload model to server
function update! end

# Client download model to server
function download end

include("fedAvgCleint.jl")

end




