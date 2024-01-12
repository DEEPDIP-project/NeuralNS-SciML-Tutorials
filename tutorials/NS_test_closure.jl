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

include("NS_functions/functions_force.jl")
include("NS_functions/functions_time.jl")
include("NS_functions/functions_plotting.jl")
include("NS_functions/functions_utils.jl")
include("NS_functions/functions_initialization.jl")
include("NS_functions/functions_params.jl")
include("NS_functions/functions_FNO.jl")
include("NS_functions/functions_CNN.jl")
include("NS_functions/functions_loss.jl")
include("NS_functions/functions_data.jl")
include("NS_functions/functions_NODE.jl")


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


# Select the parameters that identify the model you want to test
ch_fno = [2, 5, 5, 5, 2]
kmax_fno = [8, 8, 8, 8]
σ_fno = [gelu, gelu, gelu, identity]
model_name = generate_FNO_name(ch_fno, kmax_fno, σ_fno)
# including the parameters for the loss function
nunroll = 20
loss_name = "lossDtO-nu$(nunroll)"
nuse = 50
loss_name = "lossPrior-nu$(nuse)"

# Check if the model has been trained already
if isfile("trained_models/$(model_name)_$(loss_name)_$(data_name).jld2")
    print("Loading the model")
    _closure = create_fno_model(kmax_fno, ch_fno, σ_fno; single_timestep=true)
    _model_closed = create_node(_closure, simulation_data.params_les; is_closed=true)
    _model_open = create_node(_closure, simulation_data.params_les; is_closed=false)
    θ, st = Lux.setup(rng, _model_closed)
    result_training = load("trained_models/$(model_name)_$(loss_name)_$(data_name).jld2", "data")
    θ = result_training.θ
else
    throw(DomainError("The model does not exhist."))
end


# Define the NeuralODE problems (closed and open)
dt = 2f-4
tspan = (0.0f0, 1000*dt)
node_closed = NeuralODE(_model_closed, tspan, Tsit5(), adaptive=false, dt=dt, saveat=10*dt)
node_open = NeuralODE(_model_open, tspan, Tsit5(), adaptive=false, dt=dt, saveat=10*dt)



# Once trained, we can  see the closure model in action.
u₀ = dropdims(ArrayType(simulation_data.v[:, :, :, 1:1]), dims=4)
v = simulation_data.v
# run the LES with no model
open_solution = Array(node_open(u₀, θ, st)[1])
# and with the closure
closed_solution = Array(node_closed(u₀, θ, st)[1])


# Plot the comparison
anim = Animation()
# at the same time measure the error
tot_error_open = 0;
tot_error_closed = 0;
for idx in 1:size(closed_solution, 4)
    N = simulation_data.params_les.N
    v_closed = closed_solution[:,:,:,idx]
    v_open = open_solution[:,:,:,idx]
    idx_full = 10*idx
    # if it is the last iteration, set idx_full to the last one
    if idx_full > size(v, 4)
        idx_full = size(v, 4)
    end
    tot_error_closed += sum(abs2, v_closed - v[:,:,:,idx_full]) / sum(abs2, v[:,:,:,idx_full]) 
    tot_error_open += sum(abs2, v_open - v[:,:,:,idx_full]) / sum(abs2, v[:,:,:,idx_full])
    ω = reshape(Array(vorticity(ArrayType(v[:, :, :, idx_full]), simulation_data.params_les)), N, N)
    ω_nomodel = reshape(Array(vorticity(v_open, simulation_data.params_les)), N, N)
    ω_model = reshape(Array(vorticity(v_closed, simulation_data.params_les)), N, N)
    plot_title = @sprintf("Vorticity, t = %.3f", idx_full*dt)
    fig = plot(
        heatmap(ω'; colorbar = false, title = "Target DNS"),
#        heatmap(ω_nomodel'-ω'; colorbar = false, title = "No closure"),
        heatmap(ω_model'; colorbar = false, title = "Trained LES"),
        heatmap(ω_model'-ω_nomodel'; colorbar = false, title = "NN closure");
        plot_title,
        layout = (1, 3)
    )
    frame(anim, fig)
end
gif(anim)
println(" * Total error with no closure: $(tot_error_open)\n * Total error with closure: $(tot_error_closed)")

# And save the animation
if !isdir("plots")
    mkdir("plots")
end
gif(anim, "plots/$(model_name)_$(loss_name)_$(data_name).gif", fps = 10)