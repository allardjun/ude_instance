# Simulate and then learn ARBITRARY FUNCTION using a Neural Net and 5-order polynomial, plus a real variable parameter
# Example system: the Lotka-Volterra 2-system of ODEs
# Using the DiffEqFlux packages
#
# edited and commented by Jun (jun.allard@uci.edu) based on the Julia demo at:
# https://github.com/ChrisRackauckas/universal_differential_equations
# /LotkaVolterra/scenario_1.jl

# -------------------------------------------------------


## Environment and packages
cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate()

using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, Optim
using DiffEqFlux, Flux
using Plots
gr()
using JLD2, FileIO
using Statistics
# Set a random seed for reproduceable behaviour
using Random
Random.seed!(1234)

#### NOTE
# Since the recent release of DataDrivenDiffEq v0.6.0 where a complete overhaul of the optimizers took
# place, SR3 has been used. Right now, STLSQ performs better and has been changed.

# Create a name for saving ( basically a prefix )
svname = "Scenario_1_"

# -------------------------------------------------------------------------------
# Ground truth and sample generation
# -------------------------------------------------------------------------------

## Data generation
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α*u[1] - β*u[2]*u[1]
    du[2] = γ*u[1]*u[2]  - δ*u[2]
end

# Define the experimental parameter
tspan = (0.0f0,3.0f0)
u0 = Float32[0.44249296,4.6280594]
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0,tspan, p_)
solution = solve(prob, Vern7(), abstol=1e-12, reltol=1e-12, saveat = 0.1)

# Ideal data
X = Array(solution)
t = solution.t

# Add noise in terms of the mean
x̄ = mean(X, dims = 2)
noise_magnitude = Float32(5e-2)
Xₙ = X .+ (noise_magnitude*x̄) .* randn(eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

savefig(joinpath(pwd(), "plots", "$(svname)01GroundTruth.pdf"))


# -------------------------------------------------------------------------------
# Set up the neural net model to train
# -------------------------------------------------------------------------------


## Define the network
# Gaussian RBF as activation
rbf(x) = exp.(-(x.^2))
# Multilayer FeedForward
U = FastChain(
    FastDense(2,5,rbf), FastDense(5,5, rbf), FastDense(5,5, rbf), FastDense(5,2)
)


# Get the initial parameters
p = initial_params(U)

# Define the hybrid model
function ude_dynamics!(du,u, p, t, p_true)
    û = U(u, p) # Network prediction
    du[1] =  p_true[1]*u[1] + û[1]
    du[2] = -p_true[4]*u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du,u,p,t) = ude_dynamics!(du,u,p,t,p_) #J: p_ is defined above. Contains true params.
# Define the problem
prob_nn = ODEProblem(nn_dynamics!,Xₙ[:, 1], tspan, p)

## Function to train the network
# Define a predictor
function predict(θ, X = Xₙ[:,1], T = t)
    Array(solve(prob_nn, Vern7(), u0 = X, p=θ,
                tspan = (T[1], T[end]), saveat = T,
                abstol=1e-6, reltol=1e-6,
                sensealg = ForwardDiffSensitivity()
                ))
end

# Simple L2 loss
function loss(θ)
    X̂ = predict(θ)
    sum(abs2, Xₙ .- X̂)
end

# -------------------------------------------------------------------------------
# Set up the training process
# -------------------------------------------------------------------------------

# Container to track the losses
losses = Float32[]

# Callback to show the loss during training
callback(θ,l) = begin
    push!(losses, l)
    if length(losses)%50==0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    false
end

# -------------------------------------------------------------------------------
# Training!
# -------------------------------------------------------------------------------
print("Training begins!\n")
# Jun notes: Note that there are two steps.
# The first learns the arbirary function at the values of (u_1(t),u_2(t)) that occur in the data.
# This uses the package Flux, and the function is a neural network.
# The second extends the arbitrary function to be valid over a whole region of the plane (i.e., a range of values of u_1 and u_2)
# This is done using Sparse Regression by the package DataDrivenDiffEq, and the resulting function is a 5th-order polynomial,


# First train with ADAM for better convergence -> move the parameters into a
# favourable starting positing for BFGS
res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.1f0), cb=callback, maxiters = 200)
println("Training loss after $(length(losses)) iterations: $(losses[end])")
# Train with BFGS
res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01f0), cb=callback, maxiters = 10000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")


print("Training done!\n")


# -------------------------------------------------------------------------------
# Analysis of the neural net learning
# -------------------------------------------------------------------------------

# Plot the losses
pl_losses = plot(1:200, losses[1:200], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
plot!(201:length(losses), losses[201:end], yaxis = :log10, xaxis = :log10, xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red)
savefig(pl_losses, joinpath(pwd(), "plots", "$(svname)02losses.pdf"))
# Rename the best candidate
p_trained = res2.minimizer


## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):mean(diff(solution.t))/2:last(solution.t)
X̂ = predict(p_trained, Xₙ[:,1], ts)
# Trained on noisy data vs real solution
pl_trajectory = plot(ts, transpose(X̂), xlabel = "t", ylabel ="x(t), y(t)", color = :red, label = ["UDE Approximation" nothing])
scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
savefig(pl_trajectory, joinpath(pwd(), "plots", "$(svname)03trajectory_reconstruction.pdf"))

# Ideal unknown interactions of the predictor
Ȳ = [-p_[2]*(X̂[1,:].*X̂[2,:])';p_[3]*(X̂[1,:].*X̂[2,:])']
# Neural network guess
Ŷ = U(X̂,p_trained)

pl_reconstruction = plot(ts, transpose(Ŷ), xlabel = "t", ylabel ="U(x,y)", color = :red, label = ["UDE Approximation" nothing])
plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])
savefig(pl_reconstruction, joinpath(pwd(), "plots", "$(svname)04_missingterm_reconstruction.pdf"))

# Plot the error
pl_reconstruction_error = plot(ts, norm.(eachcol(Ȳ-Ŷ)), yaxis = :log, xlabel = "t", ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = plot(pl_reconstruction, pl_reconstruction_error, layout = (2,1))
savefig(pl_missing, joinpath(pwd(), "plots", "$(svname)05_missingterm_reconstruction_and_error.pdf"))
pl_overall = plot(pl_trajectory, pl_missing)
savefig(pl_overall, joinpath(pwd(), "plots", "$(svname)06_reconstruction.pdf"))
## Symbolic regression via sparse regression ( SINDy based )


# -------------------------------------------------------------------------------
# Why are we not done yet? That was just U(x,y) at the sampled points.
# So now we need to extend to unsampled regions of x-y plane with basis.
# -------------------------------------------------------------------------------
println("Begin constructing the learned function over the u1u2 region")

# Create a Basis
@variables u[1:2]
# Generate the basis functions, multivariate polynomials up to deg 5
# and sine
b = [polynomial_basis(u, 5); sin.(u)]
basis = Basis(b, u)

# Create the thresholds which should be used in the search process
λ = Float32.(exp10.(-7:0.1:0))
# Create an optimizer for the SINDy problem
opt = STLSQ(λ)
# Define different problems for the recovery
#full_problem  = ContinuousDataDrivenProblem(solution) # fine-grain solution (i.e., cheating)
#ideal_problem = ContinuousDataDrivenProblem(X̂, ts, DX = Ȳ) # known solutions (i.e., cheating)
nn_problem    = ContinuousDataDrivenProblem(X̂, ts, DX = Ŷ)
# Test on ideal derivative data for unknown function ( not available )
println("Sparse regression")
#full_res  = solve(full_problem,  basis, opt, maxiter = 10000, progress = true)
#ideal_res = solve(ideal_problem, basis, opt, maxiter = 10000, progress = true)
nn_res    = solve(nn_problem,    basis, opt, maxiter = 10000, progress = true)
# Store the results
#results = [full_res; ideal_res; nn_res]
# Show the results
#map(println, results)
# Show the results
#map(println ∘ result, results)
# Show the identified parameters
#map(println ∘ parameter_map, results)

# -------------------------------------------------------------------------------
# Here is the learned model
# -------------------------------------------------------------------------------

# Define the recovered, hyrid model
function recovered_dynamics!(du,u, p, t)
    û = nn_res(u, p) # Network prediction
    du[1] =  p_[1]*u[1] + û[1]
    du[2] = -p_[4]*u[2] + û[2]
end


# -------------------------------------------------------------------------------
# Simulate the learned model for a long time (t=10, when it was only trained to data up to time t=3)
# -------------------------------------------------------------------------------
## Simulation

# Look at long term prediction
t_long = (0.0f0, 10.0f0)
#estimation_prob = ODEProblem(estimated_dynamics!, u0, t_long, p̂)
estimation_prob = ODEProblem(recovered_dynamics!, u0, t_long, parameters(nn_res))
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1) # Using higher tolerances here results in exit of julia
plot(estimate_long)

true_prob = ODEProblem(lotka!, u0, t_long, p_)
true_solution_long = solve(true_prob, Tsit5(), saveat = estimate_long.t)
plot!(true_solution_long)

## Save the results
save(joinpath(pwd(), "results" ,"$(svname)recovery_$(noise_magnitude).jld2"),
    "solution", solution, "X", Xₙ, "t" , ts, "neural_network" , U, "initial_parameters", p, "trained_parameters" , p_trained, # Training
    "losses", losses, "result", nn_res, "recovered_parameters", parameters(nn_res), # Recovery
    #"long_solution", true_solution_long, "long_estimate", estimate_long
    ) # Estimation


## Post Processing and Plots

c1 = 3 # RGBA(174/255,192/255,201/255,1) # Maroon
c2 = :orange # RGBA(132/255,159/255,173/255,1) # Red
c3 = :blue # RGBA(255/255,90/255,0,1) # Orange
c4 = :purple # RGBA(153/255,50/255,204/255,1) # Purple

# p1 = plot(t,abs.(Array(solution) .- estimate)' .+ eps(Float32),
#           lw = 3, yaxis = :log, title = "Timeseries of UODE Error",
#           color = [3 :orange], xlabel = "t",
#           label = ["x(t)" "y(t)"],
#           #titlefont = "Helvetica", legendfont = "Helvetica",
#           legend = :topright)

# Plot L₂
p2 = plot3d(X̂[1,:], X̂[2,:], Ŷ[2,:], lw = 3,
     title = "Neural Network Fit of U2(t)", color = c1,
     label = "Neural Network", xaxis = "x", yaxis="y")
    #  titlefont = "Helvetica", legendfont = "Helvetica",
    #  legend = :bottomright)
plot!(X̂[1,:], X̂[2,:], Ȳ[2,:], lw = 3, label = "True Missing Term", color=c2)

p3 = scatter(solution, color = [c1 c2], label = ["x data" "y data"],
             title = "Extrapolated Fit From Short Training Data",
             #titlefont = "Helvetica", legendfont = "Helvetica",
             markersize = 5)

plot!(p3,true_solution_long, color = [c1 c2], linestyle = :dot, lw=5, label = ["True x(t)" "True y(t)"])
plot!(p3,estimate_long, color = [c3 c4], lw=1, label = ["Estimated x(t)" "Estimated y(t)"])
plot!(p3,[2.99,3.01],[0.0,10.0],lw=1,color=:black, label = nothing)
#annotate!([(1.5,13,text("Training \nData", 10, :center, :top, :black, "Helvetica"))])
l = @layout [grid(1,2)
             grid(1,1)]
plot(p2,p3)#,layout = l)

savefig(joinpath(pwd(),"plots","$(svname)07full_plot.pdf"))
