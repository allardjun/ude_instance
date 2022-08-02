# http://lux.csail.mit.edu/stable/examples/generated/beginner/PolynomialFitting/main/

import Lux
import NNlib, Optimisers, Plots, Random, Statistics, Zygote


function generate_data(rng::Random.AbstractRNG)
    x = reshape(collect(range(-2.0f0, 2.0f0, 128)), (1, 128))
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, 128)) .* 0.1f0
    return (x, y)
end



rng = Random.MersenneTwister()
Random.seed!(rng, 12345)

(x, y) = generate_data(rng)

Plots.plot(x -> evalpoly(x, (0, -2, 1)), x[1, :]; label=false)
Plots.scatter!(x[1, :], y[1, :]; label=false, markersize=3)

function construct_model()
    return Lux.Chain(Lux.Dense(1, 16, NNlib.relu), Lux.Dense(16, 1))
end

model = construct_model()

opt = Optimisers.Adam(0.03)

function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    mse_loss = Statistics.mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end


tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)

vjp_rule = Lux.Training.ZygoteVJP()

function main(tstate::Lux.Training.TrainState, vjp::Lux.Training.AbstractVJP, data::Tuple,
    epochs::Int)
    data = data .|> Lux.gpu
    for epoch in 1:epochs
        grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_function,
                                                                data, tstate)
        @info epoch=epoch loss=loss
        tstate = Lux.Training.apply_gradients(tstate, grads)
    end
    return tstate
end

tstate = main(tstate, vjp_rule, (x, y), 250)
y_pred = Lux.cpu(Lux.apply(tstate.model, Lux.gpu(x), tstate.parameters, tstate.states)[1])


Plots.plot(x -> evalpoly(x, (0, -2, 1)), x[1, :]; label=false)
Plots.scatter!(x[1, :], y[1, :]; label="Actual Data", markersize=3)
Plots.scatter!(x[1, :], y_pred[1, :]; label="Predictions", markersize=3)


# evaluate at a point 

println(Lux.cpu(Lux.apply(tstate.model, Lux.gpu([0]), tstate.parameters, tstate.states)[1]))
