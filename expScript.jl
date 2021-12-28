include("FedDCD.jl")
include("test/tests.jl")

# Softmax for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.3
localLr = 1e-2
numRounds = 100
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
    "results/exp1/FedAvg_logReg_mnist_lambda1e-2.txt"
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
    "results/exp1/FedProx_logReg_mnist_lambda1e-2.txt"
)

# - Scaffold
TestScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp1/Scaffold_logReg_mnist_lambda1e-2.txt"
)


# Softmax for rcv1
fileTrain = "data/rcv1_train.binary"
fileTest = "data/rcv1_train.binary"
λ = 1e-3
participationRate = 0.3
localLr = 1e-1
numRounds = 100
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
    "results/exp1/FedAvg_logReg_rcv1_lambda1e-2.txt"
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
    "results/exp1/FedProx_logReg_rcv1_lambda1e-2.txt"
)

# - Scaffold
TestScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp1/Scaffold_logReg_rcv1_lambda1e-2.txt"
)

# Softmax + MLP for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.3
localLr = 1e-2
numRounds = 100
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
    "results/exp1/FedAvg_MLP_mnist_lambda1e-2.txt"
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
    "results/exp1/FedProx_MLP_mnist_lambda1e-2.txt"
)

# - Scaffold
TestScaffoldNN(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp1/Scaffold_MLP_mnist_lambda1e-2.txt"
)