using Printf
using LinearAlgebra
using SparseArrays
using Random
using Flux

# Testing FedDCD
# TestFedDCD("data/mnist.scale", "data/mnist.scale.t")
# TestFedDCD("data/rcv1_train.multiclass", "data/rcv1_train.multiclass")
# TestFedDCD("data/covtype.scale01", "data/covtype.scale01")
function TestFedDCD(
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
    clients = Vector{FedDCDClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedDCDClient(i, Xsplit[i], Ysplit[i], clientConfig, newton!)
    end
    # Construct server
    server = FedDCDServer(Xtest, Ytest, serverConfig)

    # Train
    W, objList, testAccList = fedDCD(server, clients, numRounds)

    writeToFile(
        "covtype",
        "softmax classification",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        "results/FedDCD_logReg_COV_lambda1e-2.csv"    # file stored.
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

# W = TestFedDCDNN("data/mnist.scale", "data/mnist.scale.t");
function TestFedDCDNN(
    fileTrain::String,
    fileTest::String
    )
    numClients = 10
    numRounds = 100
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

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => 1e-3,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => 0.3,
        "learning_rate" => 1e-3,
    )
    
    # model structure
    dim = 32

    # Construct clients
    clients = Vector{FedDCDClientNN}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedDCDClientNN(i, Xsplit[i], Ysplit[i], dim, clientConfig, adam!)
    end

    # test adam!
    # adam!(clients[1].XtrainT, clients[1].Ytrain, clients[1].W, clients[1].y, clients[1].λ)

    # Construct server
    server = FedDCDServerNN(Xtest, Ytest, dim, serverConfig)

    # # Train
    W, objList, testAccList = fedDCD(server, clients, numRounds)
    @printf("Test finished!\n")

    writeToFile(
        "mnist",
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        "results/FedDCD_MLP_lambda1e-3.csv"    # file stored.
    )
    return W
end

# test neural network
# model = TestNN("data/mnist.scale", "data/mnist.scale.t");
function TestNN(
    fileTrain::String,
    fileTest::String
    )
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
    XtrainT = copy(Xtrain')
    XtestT = copy(Xtest')
    Ytrain = Flux.onehotbatch(Ytrain, 1:10)
    Ytest =  Flux.onehotbatch(Ytest, 1:10)
    # build model
    model = Chain( Dense(780, 32, relu, bias=false), Dense(32, 10, bias=false), NNlib.softmax)
    # data loader
    data = Flux.Data.DataLoader((XtrainT, Ytrain), batchsize=128, shuffle=true)
    # optimizer
    # opt = ADAMW(0.001, (0.89, 0.995), 1e-8)
    opt = Descent(1e-2)
    # loss 
    λ = 1e-2
    sqnorm(w) = sum(abs2, w)
    loss(x, l) = Flux.crossentropy(model(x), l) + (λ/2)*sum(sqnorm, params(model)) 
    #training 
    num_epoches = 100
    for t = 1:num_epoches
        @printf "epoch: %d, training obj: %.2f, test accuracy: %.2f\n" t loss(XtrainT, Ytrain) accuracy(XtestT, Ytest, model)
        Flux.train!(loss, params(model), data, opt)  
    end
    return model
end

# W = TestAccFedDCDNN("data/mnist.scale", "data/mnist.scale.t");
function TestAccFedDCDNN(
    fileTrain::String,
    fileTest::String
    )
    numClients = 10
    numRounds = 100
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

    # Setup config
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => 1e-2,
        "participation_rate" => 0.3,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => 0.3,
        "learning_rate" => 1e-2,
    )
    
    # model structure
    dim = 32

    # Construct clients
    clients = Vector{AccFedDCDClientNN}(undef, numClients)
    for i = 1:numClients
        clients[i] = AccFedDCDClientNN(i, Xsplit[i], Ysplit[i], dim, clientConfig, adam!)
    end

    # Construct server
    server = AccFedDCDServerNN(Xtest, Ytest, dim, serverConfig)

    # # Train
    W, objList, testAccList = accfedDCD(server, clients, numRounds)
    @printf("Test finished!\n")

    writeToFile(
        "mnist",
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        "results/AccFedDCD_MLP_lambda1e-2.csv"    # file stored.
    )
    return W
end