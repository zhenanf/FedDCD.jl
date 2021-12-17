########################################################################
# Training
########################################################################

# Implementation of the FedAvg and FedProx algorithm
function fedAvgAndProx(
    server::FedProxServer,
    clients::Vector{FedProxClient},
    numRounds::Int64
)
    # Connect clients with server
    server.clients = clients
    # Training process
    objList = zeros(Float64, 0)
    testAccList = zeros(Float64, 0)
    startTime = time()
    @printf("Start training!\n")
    for t = 1:numRounds
        select!(server)
        sendModel!(server)
        # objValue = getObjValue(server)
        # @printf("Round : %4d, obj: %6.4e\n", t, objValue)
        for idx in server.selectedIndices
            client = server.clients[idx]
            update!(client)
        end
        aggregate!(server)
        # Print log
        sendModelToAllClients!(server)
        objValue = getObjValue(server)
        acc = accuracy(server.Xtest, server.Ytest, server.W)
        @printf("Round : %4d, obj: %6.4e, acc: % 3.2f %%, time: %4.3f s\n", t, objValue, acc*100, time()-startTime)
        push!(objList, objValue)
        push!(testAccList, acc)
    end
    endTime = time()
    @printf("Finished training, time elapsed: %.4e\n", endTime - startTime)
    return server.W, objList, testAccList
end

# Implementation of the FedDCD algorithm
function fedDCD(
    server::FedDCDServer,
    clients::Vector{FedDCDClient},
    numRounds::Int64
)
    # Connect clients with server
    server.clients = clients
    # Training process
    startTime = time()
    @printf("Start training!\n")
    for t = 1:numRounds
        @printf("Round %d\n", t)
        select!(server)
        for idx in server.selectedIndices
            client = server.clients[idx]
            update!(client)
        end
        aggregate!(server)
        sendModel!(server)
        # Print log
        objValue = obj(server.Xtest, server.Ytest, server.W, clients[1].λ)
        acc = accuracy(server.Xtest, server.Ytest, server.W)
        @printf("Round : %4d, obj: %6.4e, acc: % 3.2f %%\n", t, objValue, acc*100)
    end
    endTime = time()
    @printf("Finished training, time elapsed: %.4e\n", endTime - startTime)
    return server.W
end

