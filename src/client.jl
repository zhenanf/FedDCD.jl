########################################################################
# Client
########################################################################

mutable struct Client{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}}
    id::T1                                  # client index
    Xtrain::T3                              # training data
    Ytrain::T5                              # training label
    w::T4                                   # (model) primal variable
    y::T4                                   # (model) dual variable
    function Client(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String, Union{Int64, Float64}})
        num_classes = config["num_classes"]
        learning_rate = config["learning_rate"]
        dm = size(Xtrain, 1)
        w = zeros(Float64, num_classes, dm)
        y = zeros(Float64, num_classes, dm)
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}}(id, Xtrain, Ytrain, w, y)
    end
end



