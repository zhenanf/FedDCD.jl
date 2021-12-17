using Printf
using LinearAlgebra
using SparseArrays
using Random
# Testing FedDCD
function testFedDCD(
    filename::String,
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 10
    numRounds = 100
    # Read data
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
        "learning_rate" => 1e-3,
        "participation_rate" => 0.3,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => 0.3,
        "learning_rate" => 1.0,
    )

    # Construct clients
    clients = Vector{FedDCDClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedDCDClient(i, Xsplit[i], ysplit[i], config, newton!)
    end
    # Construct server
    server = FedDCDServer(X, y, serverConfig)

    # Train
    W = fedDCD(server, clients, numRounds)


    @printf("Test finished!\n")
end


# Testing accelerated FedDCD
function testAccFedDCD(
    filename::String
    )
    numClients = 10
    numRounds = 100
    # Read data
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
        "learning_rate" => 1e-3,
        "participation_rate" => 0.3,
    )

    serverConfig = Dict(
        "num_classes" => numClasses,
        "num_clients" => numClients,
        "participation_rate" => 0.3,
        "learning_rate" => 1.0,
    )

    # Construct clients
    clients = Vector{AccFedDCDClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = AccFedDCDClient(i, Xsplit[i], ysplit[i], config, newton!)
    end
    # Construct server
    server = AccFedDCDServer(X, y, serverConfig)

    # Train
    W = accfedDCD(server, clients, numRounds)


    @printf("Test finished!\n")
end
