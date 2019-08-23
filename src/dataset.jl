const download = MLDatasets.MNIST.download

# TODO Mimic MLDatasets API more closely.
function traindata(; dir = nothing)
    X, T = MLDatasets.MNIST.traindata(dir = dir)
    rows, cols, n = size(X)
    (reshape(X, (rows * cols, n)), T)
end

function traindata(filtering_labels::Vector{Int}; dir = nothing)
    filter_data(traindata(dir = dir)..., filtering_labels)
end

# TODO Mimic MLDatasets API more closely.
function testdata(; dir = nothing)
    X, T = MLDatasets.MNIST.testdata(dir = dir)
    rows, cols, n = size(X)
    (reshape(X, (rows * cols, n)), T)
end

function testdata(filtering_labels::Vector{Int}; dir = nothing)
    filter_data(testdata(dir = dir)..., filtering_labels)
end

function filter_data(samples::T1, labels::T2, filtering_labels::Vector{Int}) where {T1, T2 <: AbstractArray}
    indices = findall(x -> x ∈ filtering_labels, labels)
    (samples[:,indices], labels[indices])
end
