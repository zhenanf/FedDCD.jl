push!(LOAD_PATH, pwd())
using Revise
using FedDCD
include("test/tests.jl")
include("test/fedDCDtest.jl")
