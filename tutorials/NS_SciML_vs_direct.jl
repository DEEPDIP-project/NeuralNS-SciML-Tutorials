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

include("extra_functions/functions_force.jl")
include("extra_functions/functions_time.jl")
include("extra_functions/functions_plotting.jl")
include("extra_functions/functions_utils.jl")
include("extra_functions/functions_initialization.jl")
include("extra_functions/functions_params.jl")
include("extra_functions/functions_NODE.jl")


# Lux likes to toss random number generators around, for reproducible science
rng = Random.default_rng()

# This line makes sure that we don't do accidental CPU stuff while things
# should be on the GPU
CUDA.allowscalar(false)



#
# ## Example simulation
#
#
## Initial conditions
params = create_params(64; nu = 0.001f0);
u_initial = random_field(params)
u = u_initial;

# We can also check that `u` is indeed divergence free
maximum(abs, params.k .* u[:, :, 1] .+ params.k' .* u[:, :, 2])

# Let's do some time stepping.
t = 0.0f0
dt = 1.0f-3

# Store the direct solution
W = []
ω = Array(vorticity(u, params))
push!(W, ω)
for i = 1:1000
    global u, t
    t += dt
    u = step_rk4(u, params, dt)
    if i % 10 == 0
        ω = Array(vorticity(u, params))
        push!(W, ω)
    end
end



# Test SCIML
u0 = u_initial
tspan = (0.0f0, 1.0f0)
f(u, p, t) = project(F(u, p), p)
prob = ODEProblem(f, u0, tspan, params)
# We solve using Runge-Kutta 4 to compare with the direct implementation
sol = solve(prob, RK4(), adaptive=false, dt=dt,  saveat=0.01)
# However, notice that the Tsit5() method is better, and it is straightworward to implement using SciML
#sol = solve(prob, Tsit5(), adaptive=false, dt=dt,  saveat=0.01)

# Now compare SciML with the direct implementation
anim = Animation()
fig = plot(layout = (1, 2), size = (800, 400))
@gif for (idx,(t, u)) in enumerate(zip(sol.t, sol.u))
    ω = Array(vorticity(u, params))
    w = W[idx]
    title1 = @sprintf("Vorticity, SciML, t = %.3f", t)
    title2 = @sprintf("Vorticity, Direct, t = %.3f", t)
    p1 = heatmap(ω'; xlabel = "x", ylabel = "y", titlefontsize=11, title=title1)
    p2 = heatmap(w'; xlabel = "x", ylabel = "y", titlefontsize=11, title=title2)
    #plot!(p1, p2)
    fig = plot(p1, p2)
    frame(anim, fig)
end 
gif(anim, "plots/NS_SciML_vs_direct.gif")
# Not perfect 1-1 match, but close enough. 



# ********************
# Now I do a third test using a full NeuralODE layer

# Setup the model
_closure = create_test_closure(length(u0),size(u0)[1])
_model = create_node(_closure, params; is_closed=false)
θ, st = Lux.setup(rng, _model)
# how many parameters?
length(θ)

# define the NeuralODE
prob_neuralode = NeuralODE(_model, tspan, RK4(), adaptive=false, dt=dt, saveat=0.01)
# and solve it, using the zero-initialized parameters
node_solution = Array(prob_neuralode(u0, θ, st)[1])

plot(layout = (1, 3), size = (1200, 400))
@gif for idx in 1:size(node_solution, 4)
    u = node_solution[:,:,:,idx]
    ω = Array(vorticity(u, params))
    h = sol.u[idx]
    g = Array(vorticity(h, params))
    w = W[idx]
    title1 = @sprintf("Vorticity, NODE, t = %.3f", t)
    title2 = @sprintf("Vorticity, Direct, t = %.3f", t)
    title3 = @sprintf("Vorticity, SciML, t = %.3f", t)
    p1 = heatmap(ω'; titlefontsize=11, title=title1)
    p2 = heatmap(w'; titlefontsize=11, title=title2)
    p3 = heatmap(g'; titlefontsize=11, title=title3)
    plot!(p1, p2, p3)
end




# Finally, show what happens if we use a random closure
_model = create_node(_closure, params; is_closed=true)
θ, st = Lux.setup(rng, _model)
prob_neuralode = NeuralODE(_model, tspan, RK4(), adaptive=false, dt=dt, saveat=0.01)
node_random = Array(prob_neuralode(u0, θ, st)[1])

plot(layout = (1, 2), size = (800, 400))
@gif for idx in 1:size(node_random, 4)
    u = node_random[:,:,:,idx]
    ω = Array(vorticity(u, params))
    h = node_solution[:,:,:,idx]
    w = Array(vorticity(h, params))
    title1 = @sprintf("Vorticity, RandomClosure, t = %.3f", t)
    title2 = @sprintf("Vorticity, SciML, t = %.3f", t)
    p1 = heatmap(ω'; titlefontsize=11, title=title1)
    p2 = heatmap(w'; titlefontsize=11, title=title2)
    plot!(p1, p2)
end