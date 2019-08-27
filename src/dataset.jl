const download = MLDatasets.MNIST.download

function traindata(; dir = nothing)
    X, T = MLDatasets.MNIST.traindata(dir = dir)
    reshape_data(X, T)
end

function traindata(n_samples::Int; dir = nothing)
    X, T = MLDatasets.MNIST.traindata(dir = dir)
    X₀, T₀ = random_sample(n_samples, X, T)
    reshape_data(X₀, T₀)
end

function traindata(filtering_labels::Vector{Int}; dir = nothing)
    X, T = filter_data(MLDatasets.MNIST.traindata(dir = dir)..., filtering_labels)
    reshape_data(X, T)
end

function traindata(n_samples::Int, filtering_labels::Vector{Int}; dir = nothing)
    X, T = filter_data(MLDatasets.MNIST.traindata(dir = dir)..., filtering_labels)
    X₀, T₀ = random_sample(n_samples, X, T)
    reshape_data(X₀, T₀)
end

function testdata(; dir = nothing)
    X, T = MLDatasets.MNIST.testdata(dir = dir)
    reshape_data(X, T)
end

function testdata(n_samples::Int; dir = nothing)
    X, T = MLDatasets.MNIST.testdata(dir = dir)
    X₀, T₀ = random_sample(n_samples, X, T)
    reshape_data(X₀, T₀)
end

function testdata(filtering_labels::Vector{Int}; dir = nothing)
    X, T = filter_data(MLDatasets.MNIST.testdata(dir = dir)..., filtering_labels)
    reshape_data(X, T)
end

function testdata(n_samples::Int, filtering_labels::Vector{Int}; dir = nothing)
    X, T = filter_data(MLDatasets.MNIST.testdata(dir = dir)..., filtering_labels)
    X₀, T₀ = random_sample(n_samples, X, T)
    reshape_data(X₀, T₀)
end

function filter_data(samples::T1, labels::T2,
                     filtering_labels::Vector{Int}) where {T1 <: AbstractArray, T2 <: AbstractVector}
    indices = findall(x -> x ∈ filtering_labels, labels)
    (samples[:,:,indices], labels[indices])
end

function reshape_data(samples::T1, targets::T2) where {T1 <: AbstractArray, T2 <: AbstractVector}
    r, c, N = size(samples)
    (reshape(samples, (r*c, N)), vector_to_1hot(targets))
end

function random_sample(n_samples::Int, samples::T1, labels::T2) where {T1 <: AbstractArray, T2 <: AbstractVector}
    @assert n_samples <= length(labels) "Blah."
    range = length(labels)
    indices = Random.randperm(range)[1:n_samples]
    (samples[:,:,indices], labels[indices])
end

function vector_to_1hot(targets::T) where {T <: AbstractVector}
    labels = Set(targets)
    result = zeros(eltype(targets), length(labels), length(targets))
    for (i, x) in enumerate(labels)
        result[i,:] .= targets .== x
    end
    result
end
