using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots

rng = Random.default_rng()
u0 = Float32[2.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    #true_A = [-0.1 2.0; -2.0 -0.1]
    du .= 5 .- 2 .*u
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Lux.Chain(Lux.Dense(1, 50, tanh),
                  Lux.Dense(50, 1))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot = true)
  println(l)
  # plot current prediction against data
  if doplot
    plt = scatter(tsteps, ode_data[1,:], label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    display(plot(plt))
  end
  return false
end

pinit = Lux.ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=true)


# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
                                       ADAM(0.05),
                                       callback = callback,
                                       maxiters = 300)

optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        Optim.BFGS(initial_stepnorm=0.01),
                                        callback=callback,
                                        allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)


# try to extract the NN

println(Lux.cpu(Lux.apply(dudt2, Lux.gpu([0]), result_neuralode2.u, st))[1])

nn_extraction = zeros(Float64, 128)
uvalues = reshape(collect(range(0.0f0, 5f0, 128)), (1, 128))

for i = 1:128
  nn_extraction[i] = Lux.cpu(Lux.apply(dudt2, Lux.gpu([uvalues[i]]), result_neuralode2.u, st))[1][1]
end

println(nn_extraction)

plt2 = Plots.scatter(uvalues, nn_extraction'; label="Predictions", markersize=3)
Plots.scatter!(plt2, uvalues, 5 .- 2 .*uvalues)

display(plot(plt2))