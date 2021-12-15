########################################################################
# Training
########################################################################

# function feddcd(server::Server, clients::Vector{Client}, num_rounds::Int64)
#     num_clients = server.num_clients
#     participation_rate = server.participation_rate
#     num_clients_per_round = round(num_clients * participation_rate)
#     for t = 1:num_rounds
#         # select clients
#         selected_clients = sample(1:num_clients, num_clients_per_round, replace = false)
#         for i in selected_clients
#             # clients compute dual gradients and upload
#             send_dual_gradient(clients[i], server)
#         end
#         # server adjust uploaded gradients and send back
#         adjust_gradients(server, selected_clients)
#         for i in selected_clients
#             # clients update local model
#             update_model(clients[i])
#         end
#     end
# end

# Implementation of the FedAvg algorithm
function fedAvg(
    server::FedAvgServer,
    clients::Vector{FedAvgClient},
    numRounds::Int64
)
    # Connect clients with server
    server.clients = clients
    # Training process
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
        objValue = obj(server.Xtest, server.Ytest, server.W, clients[1].lambda)
        acc = accuracy(server.Xtest, server.Ytest, server.W)
        @printf("Round : %4d, obj: %6.4e, acc: % 3.2e %%\n", t, objValue, acc*100)
    end
    endTime = time()
    @printf("Finished training, time elapsed: %.4e\n", endTime - startTime)
    return server.W
end

