using Printf
using LinearAlgebra
using SparseArrays
using Random
# Run FedAvg with logistic regression
function RunFedAvgAndProx(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    mu::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    )
    numClients = 100
    numRounds = numRounds
    # Read data
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
        fileTrain,
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
    )

    @printf("Finish training!\n")
end

# Run Scaffold with logistic regression
function RunScaffold(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    )
    numClients = 100
    numRounds = numRounds
    # Read data
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
        fileTrain,
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
    )

    @printf("Finish training!\n")
end

# Run Newton method with logistic regression
function RunNewtonMethod()
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
        D = ComputeNewtonDirection( X, Xt, W, λ, g)
        # Line-search
        η = lineSearch(X, Y, D, W, g, λ)
        W .-= η*D
    end
    
end

# Run FedAvg with neural network
function RunFedAvgAndProxNN(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    mu::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    )
    numClients = 100
    numRounds = numRounds
    # Read data
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
        fileTrain,
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
    )

    @printf("Finish training!\n")
end


# Run Scaffold for neural network
function TestScaffoldNN(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    )
    numClients = 100
    numRounds = numRounds
    # Read data
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
        fileTrain,
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
    )

    @printf("Finish training!\n")
end
