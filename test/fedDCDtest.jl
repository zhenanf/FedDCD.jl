using Printf
using LinearAlgebra
using SparseArrays
using Random
# Testing FedDCD
# TestFedDCD("data/mnist.scale", "data/mnist.scale.t")
# TestFedDCD("data/rcv1_train.multiclass", "data/rcv1_train.multiclass")
function TestFedDCD(
    fileTrain::String,
    fileTest::String
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 10
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
        "lambda" => 1e-4,
        "mu" => 0,
        "learning_rate" => 1e-3,
        "participation_rate" => 0.3,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => 0.3,
        "learning_rate" => 0.99,
    )

    # Construct clients
    clients = Vector{FedDCDClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedDCDClient(i, Xsplit[i], Ysplit[i], clientConfig, newton!)
    end
    # Construct server
    server = FedDCDServer(Xtest, Ytest, serverConfig)

    # Train
    W, objList, testAccList = fedDCD(server, clients, numRounds)

    writeToFile(
        "rcv1",
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
<<<<<<< HEAD
        "results/FedDCD_logReg_RCV1_lambda1e-4.csv"    # file stored.
=======
        "results/FedDCD_logReg_lambda1e-3.csv"    # file stored.
>>>>>>> bc30a2cf88bf3a7f0c5bed140814fbc33f582767
    )

    @printf("Test finished!\n")
end


# Testing accelerated FedDCD
# TestAccFedDCD("data/mnist.scale", "data/mnist.scale.t")
function TestAccFedDCD(
    fileTrain::String,
    fileTest::String
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
        "lambda" => 1e-2,
        "mu" => 0,
        "learning_rate" => 1e-3,
        "participation_rate" => 0.3,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => 0.3,
        "learning_rate" => 0.99,
    )

    # Construct clients
    clients = Vector{AccFedDCDClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = AccFedDCDClient(i, Xsplit[i], Ysplit[i], clientConfig, newton!)
    end
    # Construct server
    server = AccFedDCDServer(Xtrain, Ytrain, serverConfig)

    # Train
    W, objList, testAccList = accfedDCD(server, clients, numRounds, objMin = 0.0)

    writeToFile(
        "mnist",
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        "results/AccFedDCD_logReg_lambda1e-2.csv"    # file stored.
    )

    @printf("Test finished!\n")
end
