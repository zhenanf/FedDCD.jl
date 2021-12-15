using Printf
using LinearAlgebra
using SparseArrays
using Random
# Testing FedAvg
# testFedAvg("data/rcv1_train.binary")
function testFedAvgAndProx(
    filename::String,
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 10
    numRounds = 100
    # Read data
    filename = "data/rcv1_train.binary"
    # filename = "data/mnist.scale"
    X, y = read_libsvm(filename);
    y = labelTransform(y)
    numClasses = length( unique(y) )
    # Split data
    Xsplit, ysplit = splitDataByRow(X, y, numClients)    

    # Setup config, running FedAvg if mu=0.
    config = Dict(
        "num_classes" => numClasses,
        "lambda" => 1e-5,
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

function SGD()
    filename = "data/mnist.scale"
    X, y = read_libsvm(filename);
    XT = copy(X')
    y = labelTransform(y)
    K = length( unique(y) )
    lr = 1e-3
    n, d = size(X)
    W = zeros(Float64, d, K)
    perm = collect(1:n)
    lambda = 1e-8

    maxEpoch = 20
    hitTime = zeros(Int64, d, K)
    for t = 1:maxEpoch
        objval = obj(X, y, W, lambda)
        acc = accuracy(X, y, W)
        @printf("Epoch : %4d, obj: %6.4e, acc: % 3.2f %%\n", t, objval, acc*100)
        shuffle!(perm)
        fill!(hitTime, 0)
        timeStep = 1
        for i in perm
            g = getStochasticGrad(XT, y, W, i)
            g = g + lambda*W
            W = W - lr.*g
            # I, J, V = findnz(g)
            # for j = 1:length(I)
            #     idx1 = I[j]
            #     idx2 = J[j]
            #     # Lazy update
            #     delay = timeStep-hitTime[idx1, idx2]
            #     W[idx1, idx2] *= (1 - lr*lambda)^delay
            #     W[idx1, idx2] -= lr*V[j]
            #     # Update hitTime to the current time
            #     hitTime[idx1, idx2] = timeStep
            # end
            # timeStep += 1
        end
        # Lazy update for staled coordinates
        # timeStep -= 1
        # for j = 1:d
        #     for k = 1:K
        #         if hitTime[j, k] < timeStep
        #             delay = timeStep-hitTime[j, k]
        #             W[j, k] *= (1 - lr*lambda)^delay
        #         end
        #     end
        # end
    end
    return W
end