# Simulate and then learn VECTOR PARAMETERS (i.e., just a bunch of real variables)
# Example system: the Lotka-Volterra 2-system of ODEs
# Using the DiffEqFlux packages
#
# edited by Jun (jun.allard@uci) based on the Julia demo at:
# https://julialang.org/blog/2019/01/fluxdiffeq/

# -------------------------------------------------------

using DifferentialEquations
using Plots
using Flux, DiffEqFlux

# Create a name for saving ( basically a prefix )
svname = "LV_learnparams"

# -------------------------------------------------------
println("Load ODE system")

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,10.0)

# -------------------------------------------------------
println("Simulate with ground truth parameters")

p_groundtruth = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p_groundtruth)

sol = solve(prob)
plot(sol)

# -------------------------------------------------------
println("Simulate discrete data with ground truth parameters")
# So it's the same as above, but fewer timepoints

prob = ODEProblem(lotka_volterra,u0,tspan,p_groundtruth)
sol = solve(prob,Tsit5(),saveat=0.1)
observations = sol[1,:] # length 101 vector -- Note this is only one of the species (the prey species, u[1])

plot(sol)
t = 0:0.1:10.0
scatter!(t,observations)

savefig(joinpath(pwd(),"plots","$(svname)01data.pdf"))

# -------------------------------------------------------
println("Set up Flux learn")

p_learned = [2.2, 1.0, 2.0, 0.4] # Initial Parameter Vector
# This variable, p_learned, will also hold the learning result
params = Flux.params(p_learned)

function predict_rd() # Our 1-layer "neural network"
  solve(prob,Tsit5(),p=p_learned,saveat=0.1)[1,:] # override with new parameters
end

# this was the loss function in the example
#loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function

loss_rd() = sum(abs2, observations .- predict_rd()) # loss function


# -------------------------------------------------------
println("Perform Flux learning")

# Container to track the losses
losses = Float32[]

data = Iterators.repeated((), 2500)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  lossvalue = loss_rd()
  println(lossvalue)
  push!(losses,lossvalue)
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=p_learned),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, params, data, opt, cb = cb)

println("Hello world!")

# -------------------------------------------------------
println("Take a look at the results. How did we do?")

print("Learned parameters: ")
print(p_learned) # display the learned parameters. How close are they to the ground truth parameters?

print("\nGround truth parameters: ")
print(p_groundtruth)
scatter!(t,observations)

savefig(joinpath(pwd(),"plots","$(svname)02learnedTimeseries.pdf"))

# plot how the losses (difference between data and learned model) shrink over time
# (or at least we hope they shrink over time!)
plot(losses,yaxis=:log)
savefig(joinpath(pwd(),"plots","$(svname)03losses.pdf"))
