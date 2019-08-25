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

function traindata(n_samples::Int, filtering_labels::Vector{Int}; dir = nothing)
    X, T = traindata(filtering_labels; dir = dir)
    random_sample(n_samples, X, T)
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

function testdata(n_samples::Int, filtering_labels::Vector{Int}; dir = nothing)
    X, T = testdata(filtering_labels; dir = dir)
    random_sample(n_samples, X, T)
end

function filter_data(samples::T1, labels::T2,
                     filtering_labels::Vector{Int}) where {T1 <: AbstractMatrix, T2 <: AbstractVector}
    indices = findall(x -> x ∈ filtering_labels, labels)
    (samples[:,indices], labels[indices])
end

function random_sample(n_samples::Int, samples::T1, labels::T2) where {T1 <: AbstractMatrix, T2 <: AbstractVector}
    @assert n_samples <= length(labels) "Blah."
    range = length(labels)
    indices = Random.randperm(range)[1:n_samples]
    (samples[:,indices], labels[indices])
end
