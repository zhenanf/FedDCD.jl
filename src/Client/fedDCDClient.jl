# Client for FedDCD with linear local model
mutable struct FedDCDClient{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}, T6<:Function} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    XtrainT::T3                             # Row copy
    Ytrain::T5                              # training label
    W::T4                                   # (model) primal variable
    y::T4                                   # (model) dual variable
    η::T2                                   # learning rate
    λ::T2                                   # L2 regularization parameter
    oracle!::T6                             # gradient oracle
    function FedDCDClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String, Real}, oracle!::Function)
        numClasses = config["num_classes"]
        λ = config["lambda"]
        η = config["learning_rate"]
        d = size(Xtrain, 2)
        W = zeros(Float64, d, numClasses)
        y = zeros(Float64, d, numClasses)
        XtrainT = copy(Xtrain')
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}, Function}(id, Xtrain, XtrainT, Ytrain, W, y, η, λ, oracle!)
    end
end

# Model update on local device
function update!(
    client::FedDCDClient
)
    client.oracle!(client.Xtrain, client.XtrainT, client.Ytrain, client.λ, client.W, client.y, client.η)
end

# Get objective value
function getObjValue(
    client::FedDCDClient
)
    objValue = obj(client.Xtrain, client.Ytrain, client.W, client.λ)
    return objValue
end

# Client for FedDCD with multilayer perceptron local model
mutable struct FedDCDClientNN{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Zygote.Params, T5<:Flux.OneHotArray, T6<:Function, T7<:Flux.Chain} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    XtrainT::T3                             # Row copy
    Ytrain::T5                              # training label
    W::T7                                   # (model) primal variable
    y::T4                                   # (model) dual variable
    λ::T2                                   # regularization parameter
    oracle!::T6                             # gradient oracle
    function FedDCDClientNN(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, dim::Int64, config::Dict{String, Real}, oracle!::Function)
        numClasses = config["num_classes"]
        λ = config["lambda"]
        Ytrain = Flux.onehotbatch(Ytrain, 1:numClasses)
        W = Chain( Dense(780, dim, relu, bias=false), Dense(dim, 10, bias=false), NNlib.softmax)
        y = deepcopy(params(W)) 
        for j = 1:length(y)
            fill!(y[j], 0.0)
        end
        XtrainT = copy(Xtrain')
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Zygote.Params, Flux.OneHotArray, Function, Flux.Chain}(id, Xtrain, XtrainT, Ytrain, W, y, λ, oracle!)
    end
end

# Model update on local device
function update!(
    client::FedDCDClientNN
)
    # @printf("Client %d locally update\n", client.id)
    client.oracle!(client.XtrainT, client.Ytrain, client.W, client.y, client.λ)
end



# Client for AccFedDCD with linear local model
mutable struct AccFedDCDClient{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Matrix{Float64}, T5<:Vector{Int64}, T6<:Function} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    XtrainT::T3                             # Row copy
    Ytrain::T5                              # training label
    W::T4                                   # (model) primal variable
    y::T4                                   # (model) dual variable 1
    z::T4                                   # (model) dual variable 2
    v::T4                                   # (model) dual variable 3
    u::T4                                   # (model) dual variable 4
    η::T2                                   # learning rate
    r::T2                                   # participation rate
    λ::T2                                   # L2 regularization parameter
    κ::T2                                   # conditional number
    a::T2                                   # hyper parameter
    b::T2                                   # hyper parameter
    oracle!::T6                             # gradient oracle
    function AccFedDCDClient(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, config::Dict{String, Real}, oracle!::Function)
        numClasses = config["num_classes"]
        λ = config["lambda"]
        η = config["learning_rate"]
        r = config["participation_rate"]
        d = size(Xtrain, 2)
        W = zeros(Float64, d, numClasses)
        y = zeros(Float64, d, numClasses)
        z = zeros(Float64, d, numClasses)
        v = zeros(Float64, d, numClasses)
        u = zeros(Float64, d, numClasses)
        XtrainT = copy(Xtrain')
        κ = λ
        a = sqrt(κ) / (1/r + sqrt(κ))
        b = a*κ*r^2
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Matrix{Float64}, Vector{Int64}, Function}(id, Xtrain, XtrainT, Ytrain, W, y, z, v, u, η, r, λ, κ, a, b, oracle!)
    end
end

# Model updates on local device
function updateW!(client::AccFedDCDClient)
    # @printf("Client %d locally update\n", client.id)
    client.oracle!(client.Xtrain, client.XtrainT, client.Ytrain, client.λ, client.W, client.v, client.η)
end

function updatev!(client::AccFedDCDClient)
    a = client.a
    client.v .= ((1-a)*client.y + a*client.z)
    client.y .= copy(client.v)
end

function updateu!(client::AccFedDCDClient)
    a = client.a; b = client.b
    θ1 = a^2 / (a^2 + b); θ2 = b / (a^2 + b)
    client.u .= (θ1*client.z + θ2*client.v)
    client.z .= copy(client.u)
end

# Client for AccFedDCD with multilayer perceptron local model
mutable struct AccFedDCDClientNN{T1<:Int64, T2<:Float64, T3<:SparseMatrixCSC{Float64, Int64}, T4<:Zygote.Params, T5<:Flux.OneHotArray, T6<:Function, T7<:Flux.Chain} <: AbstractClient
    id::T1                                  # client index
    Xtrain::T3                              # training data
    XtrainT::T3                             # Row copy
    Ytrain::T5                              # training label
    W::T7                                   # (model) primal variable
    y::T4                                   # (model) dual variable 1
    z::T4                                   # (model) dual variable 2
    v::T4                                   # (model) dual variable 3
    u::T4                                   # (model) dual variable 4
    λ::T2                                   # regularization parameter
    r::T2                                   # participation rate
    κ::T2                                   # conditional number
    a::T2                                   # hyper parameter
    b::T2                                   # hyper parameter
    oracle!::T6                             # gradient oracle
    function AccFedDCDClientNN(id::Int64, Xtrain::SparseMatrixCSC{Float64, Int64}, Ytrain::Vector{Int64}, dim::Int64, config::Dict{String, Real}, oracle!::Function)
        numClasses = config["num_classes"]
        λ = config["lambda"]
        r = config["participation_rate"]
        Ytrain = Flux.onehotbatch(Ytrain, 1:numClasses)
        W = Chain( Dense(780, dim, relu, bias=false), Dense(dim, 10, bias=false), NNlib.softmax)
        y = deepcopy(params(W))
        z = deepcopy(params(W))
        v = deepcopy(params(W))
        u = deepcopy(params(W))
        for j = 1:length(y)
            fill!(y[j], 0.0)
            fill!(z[j], 0.0)
            fill!(v[j], 0.0)
            fill!(u[j], 0.0)
        end
        XtrainT = copy(Xtrain')
        κ = 30*λ
        a = sqrt(κ) / (1/r + sqrt(κ))
        b = a*κ*r^2
        new{Int64, Float64, SparseMatrixCSC{Float64, Int64}, Zygote.Params, Flux.OneHotArray, Function, Flux.Chain}(id, Xtrain, XtrainT, Ytrain, W, y, z, v, u, λ, r, κ, a, b, oracle!)
    end
end

# Model updates on local device
function updateW!(client::AccFedDCDClientNN)
    client.oracle!(client.XtrainT, client.Ytrain, client.W, client.v, client.λ)
end

function updatev!(client::AccFedDCDClientNN)
    a = client.a
    for j = 1:length(client.v)
        client.v[j] .= (1-a)*client.y[j] + a*client.z[j]
        client.y[j] .= client.v[j]
    end
end

function updateu!(client::AccFedDCDClientNN)
    a = client.a; b = client.b
    θ1 = a^2 / (a^2 + b); θ2 = b / (a^2 + b)
    for j = 1:length(client.u)
        client.u[j] .= θ1*client.z[j] + θ2*client.v[j]
        client.z[j] .= client.u[j]
    end
end