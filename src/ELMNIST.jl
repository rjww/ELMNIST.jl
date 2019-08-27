module ELMNIST

using  ExtremeLearningMachine
import MLDatasets
import Random

include("dataset.jl")
include("model.jl")

export download, testdata, traindata,
       Linear, ReLU, Sigmoid, Square, Tanh,
       train_model, calculate_error

end # module
