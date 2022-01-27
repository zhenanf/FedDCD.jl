# FedDCD.jl
Implementation of the federated dual coordinate descent (FedDCD) method.

## Installation
To install, just call
```julia
Pkg.add("https://github.com/ZhenanFanUBC/FedDCD.jl.git")
```

## Get data
We get data from the website of [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/). To download the datasets, just call
```bash
mkdir data
cd ./data
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzip2 -d rcv1_train.binary.bz2
bzip2 -d rcv1_test.binary.bz2
bzip2 -d mnist.scale.bz2
bzip2 -d mnist.scale.t.bz2
```

## Run FedAvg for toy example.
```julia
include("experiments/PrimalMethods.jl")
RunFedAvgAndProx(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary"
    1e-2,
    0.0,
    0.3,
    0.1,
    100,
    "results/toy.txt"
    )
```

## Run FedDCD for toy example.
```julia
include("experiments/DualMethods.jl")
RunFedDCD(
    "data/rcv1_train.binary",
    "data/rcv1_train.binary"
    1e-2,
    0.3,
    0.1,
    100,
    "results/toy.txt"
    )
```

## Credits

FedDCD.jl is developed by
[Zhenan Fan](https://zhenanf.me/)
and [Huang Fang](https://www.cs.ubc.ca/~hgfang/)

