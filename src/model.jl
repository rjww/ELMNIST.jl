const SLFN = ExtremeLearningMachine.SLFN
const ActivationFunction = ExtremeLearningMachine.ActivationFunction

function train_model(n_samples::Int,
                     filtering_labels::Vector{Int},
                     n_neurons::Int,
                     activation_function::ActivationFunction)
    X, T = traindata(n_samples, filtering_labels)
    d, N = size(X)
    q = length(filtering_labels)

    elm = ELM{Float64}(d, q, n_neurons, activation_function)
    add_data!(elm, X, T)
    solve(elm)
end

function calculate_error(model::SLFN, samples::T1, truth::T2) where {T1, T2 <: AbstractMatrix}
    X = samples
    T = truth
    Y = predict(model, X)
    d, N = size(X)

    error = 1 - (sum([argmax(T[:,n]) == argmax(Y[:,n]) for n ∈ 1:N]) / N)
end
