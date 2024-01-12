# Compare the direct implementation of the model with the SciML implementation

using LinearAlgebra
using Plots
using SparseArrays
using Printf
using FFTW
using Distributions
using DifferentialEquations
using SciMLSensitivity
using DiffEqFlux
using Statistics
using Random
using LaTeXStrings

include("Adv_functions/bc.jl")
include("Adv_functions/utils.jl")
include("Adv_functions/mesh.jl")
include("Adv_functions/rhs.jl")
include("Adv_functions/initialization.jl")
include("Adv_functions/svd.jl")
include("Adv_functions/pod.jl")
include("Adv_functions/cnn.jl")
include("Adv_functions/nn.jl")
include("Adv_functions/time_integration.jl")
include("Adv_functions/loss.jl")


# Construct the grid
x= collect(LinRange(-pi,pi,101))
y = collect(LinRange(-pi,pi,101))
# unknows per cell
UPC = 1
# generate the fine mesh
fine_mesh = gen_mesh(x,y,UPC = UPC)
# then to generate the coarse mesh, we need to define the compression factor J
# which tells you how many fine cells are compressed into one coarse cell (i.e. one point every J[i])
J = (10,10)
coarse_mesh = generate_coarse_from_fine_mesh(fine_mesh,J)
# and create a structure that contains both meshes
MP = gen_mesh_pair(fine_mesh,coarse_mesh)


# *****************
# ** Set up the model (right hand side of the equation [RHS]) **
dudt = gen_conv_advection_rhs(0.1,0.00001)
#dudt = gen_conv_burgers_rhs(0.00005)

# define the right hand side from the dudt of the model
rhs(u,mesh,t;dudt = dudt,other_arguments = 0) = dudt(padding(u,(1,1),circular = true),mesh,t)


# ** Initial conditions **
max_k = 10
energy_norm = 1
number_of_simulations = 5
init_conds = generate_random_field(fine_mesh.N,max_k,norm = energy_norm,samples = (1,number_of_simulations))


# ** Generate training data **
t_start = 0
t_end = 10
dt = 0.01
save_every = 5
pre_allocate = true

t_data,sim_data = simulate(init_conds,fine_mesh,dt,t_start,t_end,rhs,time_step,save_every = save_every,pre_allocate = pre_allocate) 
# append initial condition to position 1
sim_data = cat([init_conds,sim_data]...,dims = 5)


# ** Compare the simulation with SciML 
tspan=(t_start,t_end);
u0 = init_conds;
f(u, p, t) = rhs(u,fine_mesh,t)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, RK4(), adaptive=false, dt=dt,saveat=dt*save_every)
# plot side by side
anim = Animation()
fig = plot(layout = (1, 2), size = (800, 400))
@gif for (idx,(t, u)) in enumerate(zip(sol.t, sol.u))
    dir = sim_data[:,:,1,1,idx]
    sci = u[:,:,1,1]
    title1 = @sprintf("Vorticity, SciML, t = %.3f", t)
    title2 = @sprintf("Vorticity, Direct, t = %.3f", t)
    p1 = heatmap(sci; xlabel = "x", ylabel = "y", titlefontsize=11, title=title1)
    p2 = heatmap(dir; xlabel = "x", ylabel = "y", titlefontsize=11, title=title2)
    #plot!(p1, p2)
    fig = plot(p1, p2)
    frame(anim, fig)
end 
gif(anim, "plots/NS_SciML_vs_direct.gif", fps=10)