const SLFN = ExtremeLearningMachine.SLFN
const ActivationFunction = ExtremeLearningMachine.ActivationFunction

function prepare_model(::Type{T},
                       n_neurons::Int,
                       activation_function::ActivationFunction) where {T <: Number}
    train(T, traindata()..., n_neurons, activation_function)
end

function prepare_model(::Type{T},
                       n_samples::Int,
                       n_neurons::Int,
                       activation_function::ActivationFunction) where {T <: Number}
    train(T, traindata(n_samples)..., n_neurons, activation_function)
end

function prepare_model(::Type{T},
                       filtering_labels::Vector{Int},
                       n_neurons::Int,
                       activation_function::ActivationFunction) where {T <: Number}
    train(T, traindata(filtering_labels)..., n_neurons, activation_function)
end

function prepare_model(::Type{T},
                       n_samples::Int,
                       filtering_labels::Vector{Int},
                       n_neurons::Int,
                       activation_function::ActivationFunction) where {T <: Number}
    train(T, traindata(n_samples, filtering_labels)..., n_neurons, activation_function)
end

function train(::Type{T1},
               samples::T2,
               targets::T3,
               n_neurons::Int,
               activation_function::ActivationFunction) where {T1 <: Number, T2, T3 <: AbstractMatrix}
    d, N = size(samples)
    q, _ = size(targets)
    elm = ELM{T1}(d, q, n_neurons, activation_function)
    add_data!(elm, samples, targets)
    solve(elm)
end

function calculate_error(model::SLFN, samples::T1, truth::T2) where {T1, T2 <: AbstractMatrix}
    X = samples
    T = truth
    Y = predict(model, X)
    d, N = size(X)

    error = 1 - (sum([argmax(T[:,n]) == argmax(Y[:,n]) for n ∈ 1:N]) / N)
end
