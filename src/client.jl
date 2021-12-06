########################################################################
# Client
########################################################################

mutable struct Client{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}, T6<:Function}
    id::T1                                  # client index
    Xtrain::T3                              # training data
    Ytrain::T5                              # training label
    w::T4                                   # (model) primal variable
    y::T4                                   # (model) dual variable
    oracle::T6                              # gradient oracle
    function Client(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String, Union{Int64, Float64}}, oracle::Function)
        num_classes = config["num_classes"]
        learning_rate = config["learning_rate"]
        d = size(Xtrain, 1)
        w = zeros(Float64, num_classes, d)
        y = zeros(Float64, num_classes, d)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}, Function}(id, Xtrain, Ytrain, w, y, oracle)
    end
end



