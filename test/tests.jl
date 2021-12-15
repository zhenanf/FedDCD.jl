using Printf
using LinearAlgebra
using SparseArrays
using Random
# Testing FedAvg
# testFedAvg("data/rcv1_train.binary")
function TestFedAvgAndProx(
    filename::String,
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 10
    numRounds = 100
    # Read data
    # filename = "data/rcv1_train.binary"
    # filename = "data/mnist.scale"
    X, y = read_libsvm(filename);
    y = labelTransform(y)
    numClasses = length( unique(y) )
    # Split data
    Xsplit, ysplit = splitDataByRow(X, y, numClients)    

    # Setup config, running FedAvg if mu=0.
    config = Dict(
        "num_classes" => numClasses,
        "lambda" => 1e-2,
        "mu" => 0,
        "learning_rate" => 1e-1,
        "numLocalEpochs" => 5,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => 0.3
    )

    # Construct clients
    clients = Vector{FedProxClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedProxClient(i, Xsplit[i], ysplit[i], config)
    end
    # Construct server
    server = FedProxServer(X, y, serverConfig)

    # Train
    W = fedAvgAndProx(server, clients, numRounds);


    @printf("Test finished!\n")
end

function TestNewtonMethod()
    filename = "data/mnist.scale"
    X, y = read_libsvm(filename);
    XT = copy(X')
    y = labelTransform(y)
    K = length( unique(y) )
    n, d = size(X)
    W = zeros(Float64, d, K)
    lambda = 1e-2

    W = SoftmaxNewtonMethod(X, XT, y, W, lambda)
    
end