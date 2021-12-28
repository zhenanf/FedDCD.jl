include("setup.jl")

# Softmax for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.3
localLr = 1e-2
numRounds = 100


# Experiment 2
#   rcv1
#       τ = 0.3
# TestFedDCD(
#     "data/rcv1_train.binary",
#     "data/rcv1_train.binary",
#     0.3,
#     100,
#     "results/exp2/FedDCD_rcv1_lambda1e-3_tau3e-1.txt"
# )
# TestAccFedDCD(
#     "data/rcv1_train.binary",
#     "data/rcv1_train.binary",
#     0.3,
#     100,
#     "results/exp2/AccFedDCD_rcv1_lambda1e-3_tau3e-1.txt"
# )
# #       τ = 0.1
# TestFedDCD(
#     "data/rcv1_train.binary",
#     "data/rcv1_train.binary",
#     0.1,
#     500,
#     "results/exp2/FedDCD_rcv1_lambda1e-3_tau1e-1.txt"
# )
# TestAccFedDCD(
#     "data/rcv1_train.binary",
#     "data/rcv1_train.binary",
#     0.1,
#     500,
#     "results/exp2/AccFedDCD_rcv1_lambda1e-3_tau1e-1.txt"
# )
# #       τ = 0.05
# TestFedDCD(
#     "data/rcv1_train.binary",
#     "data/rcv1_train.binary",
#     0.05,
#     1000,
#     "results/exp2/FedDCD_rcv1_lambda1e-3_tau5e-2.txt"
# )
# TestAccFedDCD(
#     "data/rcv1_train.binary",
#     "data/rcv1_train.binary",
#     0.05,
#     1000,
#     "results/exp2/AccFedDCD_rcv1_lambda1e-3_tau5e-2.txt"
# )

#   mnist
#       τ = 0.3
# TestFedDCDNN(
#     "data/mnist.scale",
#     "data/mnist.scale.t",
#     0.3,
#     100,
#     "results/exp2/FedDCD_mnist_lambda1e-2_tau3e-1.txt"
# )
# TestAccFedDCDNN(
#     "data/mnist.scale",
#     "data/mnist.scale.t",
#     0.3,
#     100,
#     "results/exp2/AccFedDCD_mnist_lambda1e-2_tau3e-1.txt"
# )
# #       τ = 0.1
# TestFedDCDNN(
#     "data/mnist.scale",
#     "data/mnist.scale.t",
#     0.1,
#     500,
#     "results/exp2/FedDCD_mnist_lambda1e-2_tau1e-1.txt"
# )
# TestAccFedDCDNN(
#     "data/mnist.scale",
#     "data/mnist.scale.t",
#     0.1,
#     500,
#     "results/exp2/AccFedDCD_mnist_lambda1e-2_tau1e-1.txt"
# )
# #       τ = 0.05
# TestFedDCDNN(
#     "data/mnist.scale",
#     "data/mnist.scale.t",
#     0.05,
#     1000,
#     "results/exp2/FedDCD_mnist_lambda1e-2_tau5e-2.txt"
# )
# TestAccFedDCDNN(
#     "data/mnist.scale",
#     "data/mnist.scale.t",
#     0.05,
#     1000,
#     "results/exp2/AccFedDCD_mnist_lambda1e-2_tau5e-2.txt"
# )

# Experiment 3
# mnist
TestFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    0.3,
    100,
    "results/exp3/FedDCD_mnist_lambda1e-2_niid.txt"
)
TestAccFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    0.3,
    100,
    "results/exp3/AccFedDCD_mnist_lambda1e-2_niid.txt"
)