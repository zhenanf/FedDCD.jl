# FedDCD.jl
Implementation of the federated dual coordinate descent (FedDCD) method.

Run the following code before any tests and experiments.
```
include("setup.jl")
```

## Run FedAvg for toy example.
```
include("test/tests.jl")
TestFedAvgAndProx("")
```

## Test Newton's method on MNIST data.

Need to dowload MNIST data to the `data` folder first. Then run the following code
```
include("test/tests.jl")
TestNewtonMethod()
```

