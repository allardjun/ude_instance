# Simulate and then learn VECTOR PARAMETERS (i.e., just a bunch of real variables)
# Example system: the Lotka-Volterra 2-system of ODEs
# Using the DiffEqFlux packages
# Jun (jun.allard@uci) edits based on the Julia demo at:
# https://julialang.org/blog/2019/01/fluxdiffeq/



# -------------------------------------------------------

using DifferentialEquations
using Plots
using Flux, DiffEqFlux

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

p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)

sol = solve(prob)
plot(sol)

# -------------------------------------------------------
println("Simulate discrete data with ground truth parameters")

p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5(),saveat=0.1)
A = sol[1,:] # length 101 vector

plot(sol)
t = 0:0.1:10.0
scatter!(t,A)


# -------------------------------------------------------
println("Set up Flux learn")

p = [2.2, 1.0, 2.0, 0.4] # Initial Parameter Vector
params = Flux.params(p)

function predict_rd() # Our 1-layer "neural network"
  solve(prob,Tsit5(),p=p,saveat=0.1)[1,:] # override with new parameters
end

# this was the loss function in the example
#loss_rd() = sum(abs2,x-1 for x in predict_rd()) # loss function

loss_rd() = sum(abs2, A .- predict_rd()) # loss function


# -------------------------------------------------------
println("Perform Flux learning")

# Container to track the losses
losses = Float32[]

data = Iterators.repeated((), 2500)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  lossvalue = loss_rd()
  display(lossvalue)
  push!(losses,lossvalue)
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.1),ylim=(0,6)))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_rd, params, data, opt, cb = cb)

# -------------------------------------------------------
println("Take a look at the results. How did we do?")

print(p) # display the learned parameters. How close are they to the ground truth parameters?
scatter!(t,A)
