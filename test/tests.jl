using Printf
using LinearAlgebra
using SparseArrays
using Random
# Testing FedAvg
# testFedAvg("data/rcv1_train.binary")
# TestFedAvgAndProx("data/")
function TestFedAvgAndProx(
    fileTrain::String,
    fileTest::String
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 100
    numRounds = 100
    # Read data
    # filename = "data/rcv1_train.binary"
    # filename = "data/mnist.scale"
    Xtrain, Ytrain = read_libsvm(fileTrain)
    Xtest, Ytest = read_libsvm(fileTest)
    Ytrain, Ytest = labelTransform(Ytrain, Ytest)
    # Set Xtrain and Xtest same number of feature
    Itr, Jtr, Vtr = findnz(Xtrain)
    Ite, Jte, Vte = findnz(Xtest)
    d = max( size(Xtrain, 2), size(Xtest, 2) )
    Xtrain = sparse(Itr, Jtr, Vtr, size(Xtrain, 1), d)
    Xtest = sparse(Ite, Jte, Vte, size(Xtest, 1), d)
    
    numClasses = length( union( Set(Ytrain), Set(Ytest) ) )
    # Split data
    Xsplit, Ysplit = splitDataByRow(Xtrain, Ytrain, numClients)    

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => 1e-3,
        "mu" => 1e-2,
        "learning_rate" => 1e-3,
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
        clients[i] = FedProxClient(i, Xsplit[i], Ysplit[i], clientConfig)
    end
    # Construct server
    server = FedProxServer(Xtest, Ytest, serverConfig)

    # Train
    _, objList, testAccList = fedAvgAndProx(server, clients, numRounds)
    writeToFile(
        "mnist",
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        "results/FedProx_logReg_lambda1e-3.csv"    # file stored.
    )
    # writeToCSV(objList, testAccList, "results/FedAvg_logReg_lambda1e-2.csv")

    @printf("Test finished!\n")
end

function TestNewtonMethod()
    fileTrain = "data/mnist.scale"
    fileTest = "data/mnist.scale.t"
    fileTrain = "data/rcv1_train.multiclass"
    fileTest = "data/rcv1_train.multiclass"
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
    λ = 1e-5

    maxIter = 20
    tol = 1e-4
    @printf("start training!\n")
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