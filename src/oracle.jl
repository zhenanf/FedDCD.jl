########################################################################
# Gradient oracles
########################################################################

# run SVRG to obtain an exact/ineact dual gradient
function svrg(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, w::Matrix{Float64}, y::Matrix{Float64})
    1
end

# run Newton's method to obtain an exact/ineact dual gradient
function newton(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, w::Matrix{Float64}, y::Matrix{Float64})
    1
end

# run SGD to obtain an exact/ineact dual gradient
function sgd(X::SparseMatrixCSC{Float64, Int64}, Y::Vector{Int64}, w::Matrix{Float64}, y::Matrix{Float64})
    1
end