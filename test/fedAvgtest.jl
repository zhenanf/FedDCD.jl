using Printf
# Testing FedAvg
# testFedAvg("data/rcv1_train.binary")
function testFedAvg(
    filename::String,
    # participationRate::Float64,
    # lambda::Float64
    )
    numClients = 10
    numRounds = 100
    # Read data
    filename = "data/rcv1_train.binary"
    X, y = read_libsvm(filename);
    y = convert( Array{Int64}, 2 .* ( y .- 1.5 ) )
    # Split data
    Xsplit, ysplit = splitDataByRow(X, y, numClients)    

    # Setup config
    config = Dict(
        "num_classes" => 2,
        "lambda" => 1e-5,
        "learning_rate" => 1e-2,
        "numLocalEpochs" => 5,
    )

    serverConfig = Dict(
        "num_classes" => 2,
        "num_clients" => numClients,
        "participation_rate" => 0.3
    )

    # Construct clients
    clients = Vector{FedAvgClient}(undef, numClients)
    for i = 1:numClients
        clients[i] = FedAvgClient(i, Xsplit[i], ysplit[i], config)
    end
    # Construct server
    server = FedAvgServer(X, y, serverConfig)

    # Train
    w = fedAvg(server, clients, numRounds);


    @printf("Test finished!\n")
end



