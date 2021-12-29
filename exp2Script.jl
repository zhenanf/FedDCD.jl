include("setup.jl")
include("test/tests.jl")
##################################
#     Participation rate = 0.1
##################################
# Softmax for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.1
localLr = 1e-3
numRounds = 500
# - FedAvg
μ = 0.0

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedAvg_logReg_mnist_lambda1e-2_tau1e-1.txt"
)

# - FedProx
μ = 1e-4

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedProx_logReg_mnist_lambda1e-2_tau1e-1.txt"
)

# - Scaffold
TestScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp3/Scaffold_logReg_mnist_lambda1e-2_tau1e-1.txt"
)


# Softmax for rcv1
fileTrain = "data/rcv1_train.binary"
fileTest = "data/rcv1_train.binary"
λ = 1e-3
participationRate = 0.1
localLr = 1e-1
numRounds = 500
# - FedAvg
μ = 0.0

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedAvg_logReg_rcv1_lambda1e-2_tau1e-1.txt"
)

# - FedProx
μ = 1e-4

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedProx_logReg_rcv1_lambda1e-2_tau1e-1.txt"
)

# - Scaffold
TestScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/Scaffold_logReg_rcv1_lambda1e-2_tau1e-1.txt"
)

# Softmax + MLP for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.1
localLr = 1e-2
numRounds = 500
# - FedAvg
μ = 0.0

# - FedAvg
TestFedAvgAndProxNN(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedAvg_MLP_mnist_lambda1e-2_tau1e-1.txt"
)

# - FedProx

μ = 1e-4
TestFedAvgAndProxNN(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedProx_MLP_mnist_lambda1e-2_tau1e-1.txt"
)

# - Scaffold
TestScaffoldNN(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/Scaffold_MLP_mnist_lambda1e-2_tau1e-1.txt"
)


#######################################################################################################
##################################
#     Participation rate = 0.05
##################################
# Softmax for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.05
localLr = 1e-3
numRounds = 1000
# - FedAvg
μ = 0.0

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedAvg_logReg_mnist_lambda1e-2_tau5e-2.txt"
)

# - FedProx
μ = 1e-4

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedProx_logReg_mnist_lambda1e-2_tau5e-2.txt"
)

# - Scaffold
TestScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp3/Scaffold_logReg_mnist_lambda1e-2_tau5e-2.txt"
)


# Softmax for rcv1
fileTrain = "data/rcv1_train.binary"
fileTest = "data/rcv1_train.binary"
λ = 1e-3
participationRate = 0.05
localLr = 1e-1
numRounds = 1000
# - FedAvg
μ = 0.0

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedAvg_logReg_rcv1_lambda1e-2_tau5e-2.txt"
)

# - FedProx
μ = 1e-4

TestFedAvgAndProx(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedProx_logReg_rcv1_lambda1e-2_tau5e-2.txt"
)

# - Scaffold
TestScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/Scaffold_logReg_rcv1_lambda1e-2_tau5e-2.txt"
)

# Softmax + MLP for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.05
localLr = 1e-2
numRounds = 1000
# - FedAvg
μ = 0.0

# - FedAvg
TestFedAvgAndProxNN(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedAvg_MLP_mnist_lambda1e-2_tau5e-2.txt"
)

# - FedProx

μ = 1e-4
TestFedAvgAndProxNN(
    fileTrain,
    fileTest,
    λ,
    μ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/FedProx_MLP_mnist_lambda1e-2_tau5e-2.txt"
)

# - Scaffold
TestScaffoldNN(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/Scaffold_MLP_mnist_lambda1e-2_tau5e-2.txt"
)