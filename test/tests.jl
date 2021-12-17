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
    y, y = labelTransform(y, y)
    numClasses = length( unique(y) )
    # Split data
    Xsplit, ysplit = splitDataByRow(X, y, numClients)    

    # Setup config, running FedAvg if mu=0.
    config = Dict(
        "num_classes" => numClasses,
        "lambda" => 1e-2,
        "mu" => 0,
        "learning_rate" => 1e-4,
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
    fileTrain = "data/mnist.scale"
    fileTest = "data/mnist.scale.t"
    X, Y = read_libsvm(fileTrain);
    Xtest, Ytest = read_libsvm(fileTest)
    Xt = copy(X')
    Y, Ytest = labelTransform(Y, Ytest)
    K = length( unique(Y) )
    n, d = size(X)
    I, J, V = findnz(Xtest)
    Xtest = sparse(I, J, V, size(Xtest,1), d)
    @show(size(X))
    @show(size(Xtest))
    W = zeros(Float64, d, K)
    λ = 1e-4

    maxIter = 20
    tol = 1e-4
    startTime = time()
    for iter = 1:maxIter
        objval = obj(X, Y, W, λ)
        g = getGradient(X, Xt, Y, W, λ)
        accTrain = accuracy(X, Y, W)
        accTest = accuracy(Xtest, Ytest, W)
        gnorm = norm(g)
        @printf("Iter %3d, obj: %4.5e, gnorm: %4.5e, train: %3.3f, test: %3.3f, time: %4.2f\n", iter, objval, gnorm, accTrain*100, accTest*100, time()-startTime)
        if gnorm < tol
            break
        end
        # Compute Newton direction.
        D = ComputeNewtonDirection2( X, Xt, Y, W, λ, g)
        # Use stepsize 1.
        W -=  D
    end
    
end