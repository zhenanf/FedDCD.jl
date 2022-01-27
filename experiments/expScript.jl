using FedDCD
include("./PrimalMethods.jl")
include("./DualMethods.jl")

# Softmax for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.3
localLr = 1e-2
numRounds = 100
# - FedAvg
μ = 0.0

RunFedAvgAndProx(
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

RunFedAvgAndProx(
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
RunScaffold(
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

RunFedAvgAndProx(
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

RunFedAvgAndProx(
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
RunScaffold(
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
RunFedAvgAndProxNN(
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
RunFedAvgAndProxNN(
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
RunScaffoldNN(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp1/Scaffold_MLP_mnist_lambda1e-2.txt"
)

# Softmax for mnist
fileTrain = "data/mnist.scale"
fileTest = "data/mnist.scale.t"
λ = 1e-2
participationRate = 0.1
localLr = 1e-3
numRounds = 500
# - FedAvg
μ = 0.0

RunFedAvgAndProx(
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

RunFedAvgAndProx(
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
RunScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/Scaffold_logReg_mnist_lambda1e-2_tau1e-1.txt"
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

RunFedAvgAndProx(
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

RunFedAvgAndProx(
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
RunScaffold(
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
RunFedAvgAndProxNN(
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
RunFedAvgAndProxNN(
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
RunScaffoldNN(
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

RunFedAvgAndProx(
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

RunFedAvgAndProx(
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
RunScaffold(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/Scaffold_logReg_mnist_lambda1e-2_tau5e-2.txt"
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

RunFedAvgAndProx(
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

RunFedAvgAndProx(
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
RunScaffold(
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
RunFedAvgAndProxNN(
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
RunFedAvgAndProxNN(
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
RunScaffoldNN(
    fileTrain,
    fileTest,
    λ,
    participationRate,
    localLr,
    numRounds,
    "results/exp2/Scaffold_MLP_mnist_lambda1e-2_tau5e-2.txt"
)

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
RunFedDCD(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary",
    1e-3,
    0.3,
    0.1,
    100,
    "results/exp2/FedDCD_rcv1_lambda1e-3_tau3e-1.txt"
)
RunAccFedDCD(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary",
    1e-3,
    0.3,
    0.1,
    100,
    "results/exp2/AccFedDCD_rcv1_lambda1e-3_tau3e-1.txt"
)
#       τ = 0.1
RunFedDCD(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary",
    1e-3,
    0.1,
    0.1,
    500,
    "results/exp2/FedDCD_rcv1_lambda1e-3_tau1e-1.txt"
)
RunAccFedDCD(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary",
    1e-3,
    0.1,
    0.1,
    500,
    "results/exp2/AccFedDCD_rcv1_lambda1e-3_tau1e-1.txt"
)
#       τ = 0.05
RunFedDCD(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary",
    1e-3,
    0.05,
    0.1,
    1000,
    "results/exp2/FedDCD_rcv1_lambda1e-3_tau5e-2.txt"
)
RunAccFedDCD(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary",
    1e-3,
    0.05,
    0.1,
    1000,
    "results/exp2/AccFedDCD_rcv1_lambda1e-3_tau5e-2.txt"
)

#   mnist
#       τ = 0.3
RunFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    1e-2,
    0.3,
    0.1,
    100,
    "results/exp2/FedDCD_mnist_lambda1e-2_tau3e-1.txt"
)
RunAccFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    1e-2,
    0.3,
    0.1,
    100,
    "results/exp2/AccFedDCD_mnist_lambda1e-2_tau3e-1.txt"
)
#       τ = 0.1
RunFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    1e-2,
    0.1,
    0.1,
    500,
    "results/exp2/FedDCD_mnist_lambda1e-2_tau1e-1.txt"
)
RunAccFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    1e-2,
    0.1,
    0.1,
    500,
    "results/exp2/AccFedDCD_mnist_lambda1e-2_tau1e-1.txt"
)
#       τ = 0.05
RunFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    1e-2,
    0.05,
    0.1,
    1000,
    "results/exp2/FedDCD_mnist_lambda1e-2_tau5e-2.txt"
)
RunAccFedDCDNN(
    "data/mnist.scale",
    "data/mnist.scale.t",
    1e-2,
    0.05,
    0.1,
    1000,
    "results/exp2/AccFedDCD_mnist_lambda1e-2_tau5e-2.txt"
)

