module ELMNIST

using  ExtremeLearningMachine
import MLDatasets
import Random

include("dataset.jl")
include("model.jl")

export ELM,
       Linear, ReLU, Sigmoid, Square, Tanh,
       add_data!, solve, predict,
       prepare_model, calculate_error,
       download, testdata, traindata

end # module
