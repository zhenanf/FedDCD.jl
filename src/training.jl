########################################################################
# Training
########################################################################

function feddcd(server::Server, clients::Vector{Client}, num_rounds::Int64)
    num_clients = server.num_clients
    participation_rate = server.participation_rate
    num_clients_per_round = round(num_clients * participation_rate)
    for t = 1:num_rounds
        # select clients
        selected_clients = sample(1:num_clients, num_clients_per_round, replace = false)
        for i in selected_clients
            # clients compute dual gradients and upload
            send_dual_gradient(clients[i], server)
        end
        # server adjust uploaded gradients and send back
        adjust_gradients(server, selected_clients)
        for i in selected_clients
            # clients update local model
            update_model(clients[i])
        end
    end
end