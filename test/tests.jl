using Printf
using LinearAlgebra
using SparseArrays
using Random
# Testing FedAvg
# testFedAvg("data/rcv1_train.binary")
# TestFedAvgAndProx("data/rcv1_train.binary", "data/rcv1_train.binary")
function TestFedAvgAndProx(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    mu::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 100
    numRounds = numRounds
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
    # Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses ) 

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
        # "mu" => 1e-4,
        # "mu" => 0.0,
        "mu" => mu,
        "learning_rate" => localLr,
        "numLocalEpochs" => 5,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate
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
        "rcv1",
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
        # "results/FedAvg_logReg_RCV1_lambda1e-3_lr1e-2.csv"    # file stored.
    )
    # writeToCSV(objList, testAccList, "results/FedAvg_logReg_lambda1e-2.csv")

    @printf("Test finished!\n")
end

# Test function for Scaffold
# TestScaffold("data/mnist.scale", "data/mnist.scale.t")
function TestScaffold(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 100
    numRounds = numRounds
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
    # Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses )  

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
        "learning_rate" => localLr,
        "numLocalEpochs" => 5,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate,
        "learning_rate" => 1.0
    )

    # Construct clients
    clients = Vector{ScaffoldClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = ScaffoldClient(i, Xsplit[i], Ysplit[i], clientConfig)
    end
    # Construct server
    server = ScaffoldServer(Xtest, Ytest, serverConfig)

    # Train
    _, objList, testAccList = Scaffold(server, clients, numRounds)
    writeToFile(
        "rcv1",
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
        # "results/Scaffold_logReg_RCV1_lambda1e-3_lr1e-1.csv"    # file stored.
    )
    # writeToCSV(objList, testAccList, "results/FedAvg_logReg_lambda1e-2.csv")

    @printf("Test finished!\n")
end

# rcv1, lambda = 1e-3, optimal obj=4.18584773e-01
function TestNewtonMethod()
    # fileTrain = "data/mnist.scale"
    # fileTest = "data/mnist.scale.t"
    fileTrain = "data/rcv1_train.binary"
    fileTest = "data/rcv1_train.binary"
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
    λ = 1e-3

    maxIter = 20
    tol = 1e-6
    @printf("start training!\n")
    startTime = time()
    for iter = 1:maxIter
        objval = obj(X, Y, W, λ)
        g = getGradient(X, Xt, Y, W, λ)
        accTrain = accuracy(X, Y, W)
        accTest = accuracy(Xtest, Ytest, W)
        gnorm = norm(g)
        @printf("Iter %3d, obj: %4.8e, gnorm: %4.5e, train: %3.3f, test: %3.3f, time: %4.2f\n", iter, objval, gnorm, accTrain*100, accTest*100, time()-startTime)
        if gnorm < tol
            break
        end
        # Compute Newton direction.
        D = ComputeNewtonDirection2( X, Xt, Y, W, λ, g)
        # Line-search
        η = lineSearch(X, Y, D, W, g, λ)
        W .-= η*D
    end
    
end

#############################################################################################################

# TestFedAvgAndProxNN("data/mnist.scale", "data/mnist.scale.t")
function TestFedAvgAndProxNN(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    mu::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 100
    numRounds = numRounds
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
    # Xsplit, Ysplit = splitDataByRow(Xtrain, Ytrain, numClients)    
    Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses )


    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
        # "mu" => 1e-4,
        # "mu" => 0.0,
        "mu" => mu,
        "learning_rate" => localLr,
        "numLocalEpochs" => 5,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate
    )

    # model structure
    dim = 32

    # Construct clients
    clients = Vector{FedProxClientNN}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedProxClientNN(i, Xsplit[i], Ysplit[i], dim, clientConfig)
    end
    # Construct server
    server = FedProxServerNN(Xtest, Ytest, dim, serverConfig)

    # Train
    _, objList, testAccList = fedAvgAndProx(server, clients, numRounds)
    writeToFile(
        "mnist",
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
        # "results/FedAvg_MLP_lambda1e-2_lr1e-2.csv"    # file stored.
    )
    # writeToCSV(objList, testAccList, "results/FedAvg_logReg_lambda1e-2.csv")

    @printf("Test finished!\n")
end


# Test function for Scaffold with neural networks
# TestScaffoldNN("data/mnist.scale", "data/mnist.scale.t")
function TestScaffoldNN(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 100
    numRounds = numRounds
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
    # Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses )

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
        "learning_rate" => localLr,
        "numLocalEpochs" => 5,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate,
        "learning_rate" => 1.0
    )

    # model structure
    dim = 32

    # Construct clients
    clients = Vector{ScaffoldClientNN}(undef, numClients)
    for i = 1:numClients
        clients[i] = ScaffoldClientNN(i, Xsplit[i], Ysplit[i], dim, clientConfig)
    end
    # Construct server
    server = ScaffoldServerNN(Xtest, Ytest, dim, serverConfig)

    # Train
    _, objList, testAccList = Scaffold(server, clients, numRounds)
    writeToFile(
        "mnist",
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
        # "results/Scaffold_MLP_lambda1e-2_lr1e-1.csv"    # file stored.
    )
    # writeToCSV(objList, testAccList, "results/FedAvg_logReg_lambda1e-2.csv")

    @printf("Test finished!\n")
end
