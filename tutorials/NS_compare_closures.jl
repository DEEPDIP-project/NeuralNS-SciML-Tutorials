using ComponentArrays
using CUDA
using FFTW
using LinearAlgebra
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Plots
using Printf
using Random
using Zygote
using DifferentialEquations
using JLD2
using SciMLSensitivity
using DiffEqFlux
using OptimizationOptimisers
using Statistics

include("extra_functions/functions_force.jl")
include("extra_functions/functions_plotting.jl")
include("extra_functions/functions_utils.jl")
include("extra_functions/functions_params.jl")
include("extra_functions/functions_FNO.jl")
include("extra_functions/functions_CNN.jl")
include("extra_functions/functions_loss.jl")
include("extra_functions/functions_data.jl")
include("extra_functions/functions_NODE.jl")


# Lux likes to toss random number generators around, for reproducible science
rng = Random.default_rng()

# This line makes sure that we don't do accidental CPU stuff while things
# should be on the GPU
CUDA.allowscalar(false)

# fix the random seed for reproducibility
Random.seed!(1234)

# Select the parameters that define the simulation you want to target
nu = 5.0f-4
les_size = 32
dns_size = 64
les_size = 64
dns_size = 128
dataseed = 1234
data_name = get_data_name(nu, les_size, dns_size, dataseed)
# If they are there load them
if isfile("data/$(data_name).jld2")
    println("Loading data from file")
    simulation_data = load("data/$(data_name).jld2","data")
else
    throw(DomainError("The data are missing."))
end
# and some global simulation parameters
dt = 2f-4
tspan = (0.0f0, 1000*dt)
u₀ = dropdims(ArrayType(simulation_data.v[:, :, :, 1:1]), dims=4)
v = simulation_data.v

# Here list the model names (+loss) that you would like to compare
model_list = nothing
model_list = [
    "FNO__2-5-5-5-2__8-8-8-8__gelu-gelu-gelu-identity_lossMulDtO-nu5-ni4",
    "FNO__2-5-5-5-2__8-8-8-8__gelu-gelu-gelu-identity_lossPrior-nu50",
    "FNO__2-5-5-5-2__8-8-8-8__gelu-gelu-gelu-identity_lossDtO-nu20",
    "FNO__2-5-5-5-2__8-8-8-8__gelu-gelu-gelu-identity_lossDtO-nu10",
    "CNN__2-2-2__2-8-8-2__leakyrelu-leakyrelu-identity__true-true-false_lossDtO-nu10",
    "CNN__2-2-2__2-8-8-2__leakyrelu-leakyrelu-identity__true-true-false_lossPrior-nu50"
]


# For every model, I want to measure the deviation from the DNS result, compare with a no closure approach 
# So first I compute the error of the LES with no closure 
_model = create_node(Dropout(0), simulation_data.params_les; is_closed=false)
θ, st = Lux.setup(rng, _model)
node = NeuralODE(_model, tspan, Tsit5(), adaptive=false, dt=dt, saveat=10*dt)
node_solution = Array(node(u₀, θ, st)[1])
elist_noclosure = compute_error(node_solution, v)

# * And then I test all the models in the list
plot()
fig = plot(; xlabel = "(10x) tsteps", title = "(Error)/(Error_NOclosure)", yaxis=:log, legendfont=6)
for model_name in model_list
    # Check if the model has been trained already
    if isfile("trained_models/$(model_name)_$(data_name).jld2")
        println("\n\n *** Loading $(model_name)")
        trained_NODE = load("trained_models/$(model_name)_$(data_name).jld2", "data")
        _closure = _get_closure(trained_NODE)
        _model = create_node(_closure, simulation_data.params_les; is_closed=true)
        θ, st = Lux.setup(rng, _model)
        θ = trained_NODE.θ
    else
        throw(DomainError("The model $(model_name) does not exhist."))
        continue
    end

    # Define the NeuralODE problem
    node = NeuralODE(_model, tspan, Tsit5(), adaptive=false, dt=dt, saveat=10*dt)
    # and run it
    println("Running the simulation")
    node_solution = Array(node(u₀, θ, st)[1])

    # Compute the error
    println("Computing the error")
    elist = compute_error(node_solution, v)
    
    # and plot
    plot!(fig, 1:length(elist), elist./elist_noclosure, label = model_name )
    display(fig)
end
# make the top y in the plot cap at 1e1, while the bottom at the predefined value
ylims!(fig, 0.95, 1.1)
# draw a horizontal y=1 line
hline!(fig, [1], label="no closure", color=:black, linestyle=:dash)

# if it does not exist, create the folder for the plots
if !isdir("plots")
    mkdir("plots")
end
savefig(fig, "plots/$(data_name)_error.png")