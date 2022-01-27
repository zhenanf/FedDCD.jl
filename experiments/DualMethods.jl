using Printf
using LinearAlgebra
using SparseArrays
using Random
using Flux

# Run FedDCD with logistic regression
function RunFedDCD(
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

    # read data
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
    # Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses)

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
        "learning_rate" => localLr,
        "participation_rate" => participationRate,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate,
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


# Run AccFedDCD with logistic regression
function RunAccFedDCD(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    )
    numClients = 100;
    numRounds = numRounds;
    # Read data
    Xtrain, Ytrain = read_libsvm(fileTrain);
    Xtest, Ytest = read_libsvm(fileTest);
    Ytrain, Ytest = labelTransform(Ytrain, Ytest);
    # Set Xtrain and Xtest same number of feature
    Itr, Jtr, Vtr = findnz(Xtrain);
    Ite, Jte, Vte = findnz(Xtest);
    d = max( size(Xtrain, 2), size(Xtest, 2) );
    Xtrain = sparse(Itr, Jtr, Vtr, size(Xtrain, 1), d);
    Xtest = sparse(Ite, Jte, Vte, size(Xtest, 1), d);
    
    numClasses = length( union( Set(Ytrain), Set(Ytest) ) );
    # Split data
    Xsplit, Ysplit = splitDataByRow(Xtrain, Ytrain, numClients) 
    # Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses) 

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
        "learning_rate" => localLr,
        "participation_rate" => participationRate,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate,
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

# Run FedDCD with neural network
function TestFedDCDNN(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    )
    numClients = 100;
    numRounds = numRounds;
    # Read data
    Xtrain, Ytrain = read_libsvm(fileTrain);
    Xtest, Ytest = read_libsvm(fileTest);
    Ytrain, Ytest = labelTransform(Ytrain, Ytest);
    # Set Xtrain and Xtest same number of feature
    Itr, Jtr, Vtr = findnz(Xtrain);
    Ite, Jte, Vte = findnz(Xtest);
    d = max( size(Xtrain, 2), size(Xtest, 2) );
    Xtrain = sparse(Itr, Jtr, Vtr, size(Xtrain, 1), d);
    Xtest = sparse(Ite, Jte, Vte, size(Xtest, 1), d);
    
    numClasses = length( union( Set(Ytrain), Set(Ytest) ) );
    # Split data
    Xsplit, Ysplit = splitDataByRow(Xtrain, Ytrain, numClients) 
    # Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses)   

    # Setup config, running FedAvg if mu=0.
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate,
        "learning_rate" => localLr,
        "decay_rate" => 0.0,
    )
    
    # model structure
    dim = 32

    # Construct clients
    clients = Vector{FedDCDClientNN}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedDCDClientNN(i, Xsplit[i], Ysplit[i], dim, clientConfig, adam!)
    end

    # Construct server
    server = FedDCDServerNN(Xtest, Ytest, dim, serverConfig)

    # # Train
    W, objList, testAccList = fedDCD(server, clients, numRounds)
    @printf("Finish training!\n")

    writeToFile(
        fileTrain,
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
    )
    return W
end

# test neural network
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
    opt = ADAM()
    # loss 
    λ = 1e-2
    sqnorm(w) = sum(abs2, w)
    loss(x, l) = Flux.crossentropy(model(x), l) + (λ/2)*sum(sqnorm, params(model)) 
    #training 
    num_epoches = 1000
    f = open("results/exp1/optval_MNIST_MLP_lambda1e-2.txt", "w")
    for t = 1:num_epoches
        # @printf "epoch: %d, training obj: %.4f, test accuracy: %.2f\n" t loss(XtrainT, Ytrain) accuracy(Xtest, Ytest, model)
        @printf(f, "epoch: %d, training obj: %.6f\n", t, loss(XtrainT, Ytrain))
        Flux.train!(loss, params(model), data, opt)  
    end
    close(f)
    return model
end

# Run AccFedDCD with neural network
function RunAccFedDCDNN(
    fileTrain::String,
    fileTest::String,
    lambda::Float64,
    participationRate::Float64,
    localLr::Float64,
    numRounds::Int64,
    writeFileName::String
    )
    numClients = 100;
    numRounds = numRounds;
    # Read data
    Xtrain, Ytrain = read_libsvm(fileTrain);
    Xtest, Ytest = read_libsvm(fileTest);
    Ytrain, Ytest = labelTransform(Ytrain, Ytest);
    # Set Xtrain and Xtest same number of feature
    Itr, Jtr, Vtr = findnz(Xtrain);
    Ite, Jte, Vte = findnz(Xtest);
    d = max( size(Xtrain, 2), size(Xtest, 2) );
    Xtrain = sparse(Itr, Jtr, Vtr, size(Xtrain, 1), d);
    Xtest = sparse(Ite, Jte, Vte, size(Xtest, 1), d);
    
    numClasses = length( union( Set(Ytrain), Set(Ytest) ) );
    # Split data
    # Xsplit, Ysplit = splitDataByRow(Xtrain, Ytrain, numClients) 
    Xsplit, Ysplit = splitDataByClass(Xtrain, Ytrain, numClients, numClasses)      

    # Setup config
    clientConfig = Dict(
        "num_classes" => numClasses,
        "lambda" => lambda,
        "participation_rate" => participationRate,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => participationRate,
        "learning_rate" => localLr,
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
        fileTrain,
        "softmax classification with MLP",
        serverConfig,
        clientConfig,
        objList,
        testAccList,
        writeFileName
    )
    return W
end
