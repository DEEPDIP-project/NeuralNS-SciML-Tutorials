using LinearAlgebra
using Plots
using SparseArrays
using NNlib

using Printf
using FFTW
using Distributions
using JLD

using DifferentialEquations
using JLD2
using SciMLSensitivity
using DiffEqFlux
using DSP
using ComponentArrays
using CUDA
using LinearAlgebra
using Lux
using LuxCUDA
using Optimisers
using OptimizationOptimisers
using Statistics


using Flux
using Random
using LaTeXStrings
using ProgressBars
using Zygote
stop_gradient(f) = f()
Zygote.@nograd stop_gradient

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

t_start = 0
t_end = 10
dt = 0.01
save_every = 5


# ** Run simulation using SciML 
tspan=(t_start,t_end);
u0 = init_conds;
f(u, p, t) = rhs(u,fine_mesh,t)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob, Tsit5() , adaptive=false, dt=dt,saveat=dt*save_every)

# unroll sol.u into a 5D array where every element is sol.u[i]
data = reshape(cat(sol.u..., dims=length(size(sol.u[1])) + 1), (size(sol.u[1])...,size(sol.u)[1]))
# reshape into a 4D array
data = reshape(data, size(data, 1), size(data, 2), size(data, 3), size(data, 4) * size(data, 5))




# *******************************
# ** Local ProperOrthogonalDecomposition (POD) **
### momentum conserving correction of data set
conserve_momentum = true

data_hat = sqrt.(MP.omega_tilde) .* data
#reshape_for_local_SVD(data_hat,MP)

# Compute and plot the singular values
POD_modes,S = carry_out_local_SVD(data_hat,MP,subtract_average = conserve_momentum)
plot(S,marker = true)

# Select how many modes to keep, including u_bar
r =5 

if conserve_momentum 
    if r >= UPC + 1
        global_POD_modes = local_to_global_modes(POD_modes[[(:) for i in 1:fine_mesh.dims]...,:,1:(r-UPC)],MP)
    else
        global_POD_modes = 0
    end
    global_POD_modes = add_filter_to_modes(global_POD_modes,MP,orthogonalize = false)
else
    global_POD_modes = local_to_global_modes(POD_modes[[(:) for i in 1:fine_mesh.dims]...,:,1:r],MP)
end

# Finally generate the projection operator
PO = gen_projection_operators(global_POD_modes,MP,uniform =false)
# In the heatmap you can visually check if the modes are orthogonal
heatmap(compute_overlap_matrix(global_POD_modes))


# ** Generate reference data **
# Use the projection operator on the rhs of the data on the fine mesh
ref_rhs_data = PO.W(rhs(data,fine_mesh,0))
# also project the data itself
ref_data = PO.W(data)
# Then compute the energy on the fine mesh
E_ref = fine_mesh.ip(data,data)[1:end]
# and the energy on the coarse mesh, using the projection operator
E_pod = coarse_mesh.ip(ref_data,ref_data,combine_channels = true)[1:end]
# This final coefficient will tend to 1 if the projection operator is orthonormal and the magnitude of data is preserved when projected onto the coarse mesh
mean(E_pod ./ E_ref)


# ** Define the neural network **
kernel_sizes = [(2,2),(1,1)]
channels = [r+1,20] # r+1 as the first channel for the rhs
strides = [(1,1),(1,1)]
B = (1,1)
boundary_padding = "c" #[["c","c"];;[0,0]]

constrain_energy =true
dissipation = true
conserve_momentum =true

model = gen_skew_NN(kernel_sizes,channels,strides,r,B,boundary_padding = boundary_padding,UPC = coarse_mesh.UPC,constrain_energy = constrain_energy,dissipation = dissipation,conserve_momentum = conserve_momentum)


# We check if the model is dissipative by plotting the energy of the rhs calculated using the NN
plot(coarse_mesh.ip(ref_data,neural_rhs(ref_data,coarse_mesh,0))[1:end])


# *******************************
# ** Train the closure models **
# We are going to compare Derivative fitting and Trajectory fitting

# ** Derivative fitting **
batchsize = 5
derivative_fitting_data_loader = Flux.Data.DataLoader((ref_data,ref_rhs_data), batchsize=batchsize,shuffle=true)
# precompile loss
sqrt.(derivative_fitting_loss(ref_data,ref_rhs_data))

# Use Adam as optimizer
opt = ADAM()

# No need to get the parameters, you can pass model to Flux
#ps = Flux.params(model.CNN,model.B_mats...)
epochs =30
losses = zeros(epochs)
epoch = 0 
# training loop
for epoch in tqdm(1:epochs)
    Flux.train!(derivative_fitting_loss,model, derivative_fitting_data_loader, opt)
    train_loss = derivative_fitting_loss(ref_data,ref_rhs_data)
    losses[epoch] = train_loss
end

# ***** There is a proble with this updated version of Flux
# ***** switch to Lux for the NN
# the code below is not working
exit()


plot(sqrt.(losses),xguidefontsize=14,yguidefontsize=14 ,dpi = 130, legend=:topright,legendfont=font(15),linewidth = 2)
xlabel!("Iteration")
ylabel!(L"\mathcal{L}")


preds = neural_rhs(ref_data,coarse_mesh,0)

select = rand(collect(1:prod(size(ref_rhs_data))),(100))
scatter(ref_rhs_data[select],preds[select])
xlabel!("true RHS")
ylabel!("predicted RHS")


# ** Trajectory fitting **
simulation_indexes = collect(1:size(sim_data)[end-1])'
simulation_indexes = cat([simulation_indexes for i in 1:size(sim_data)[end]]...,dims = fine_mesh.dims + 2)
simulation_indexes = simulation_indexes[1:end]
simulation_times = t_data[1:end]

ref_data_trajectory = reshape(ref_data,(size(ref_data)[1:coarse_mesh.dims+1]...,size(sim_data)[end-1],size(sim_data)[end]))

sim_interpolator = gen_time_interpolator(t_data,ref_data_trajectory)


# remove data that lies at the end of the simulation
buffer_dt = 0.5
select = buffer_dt .< maximum(simulation_times) .- simulation_times

traj_data = ref_data[[(:) for i in 1:coarse_mesh.dims+1]...,select]
traj_indexes = simulation_indexes[select]
traj_times = simulation_times[select]


traj_dt = 0.05
traj_steps = 10

batchsize = 20
trajectory_fitting_data_loader = Flux.Data.DataLoader((traj_data,traj_indexes,traj_times), batchsize=batchsize,shuffle=true)
    
opt = ADAM()
select = rand(collect(1:prod(size(traj_indexes))),(200))
sqrt.(trajectory_fitting_loss(traj_data[:,:,:,select],traj_indexes[select],traj_times[select]))
    
ps = Flux.params(model.CNN,model.B_mats...)
epochs =5
losses = zeros(epochs)
for epoch in tqdm(1:epochs)
    Flux.train!(trajectory_fitting_loss,ps, trajectory_fitting_data_loader, opt)
    train_loss = trajectory_fitting_loss(traj_data[:,:,:,select],traj_indexes[select],traj_times[select])
    losses[epoch] = train_loss
end
plot(sqrt.(losses))
    
    
save_skew_model(model, "my_model")
model = load_skew_model("my_model")


# ** Online testing **
max_k = 10
energy_norm = 1
number_of_simulations = 1

new_init_conds = generate_random_field(fine_mesh.N,max_k,norm = energy_norm,samples = (1,number_of_simulations))

# ** Compute reference data **
t_start = 0
t_end = 5
dt = 0.05
save_every = 1
pre_allocate = true


t,test_sim = simulate(new_init_conds,fine_mesh,dt,t_start,t_end,rhs,time_step,save_every = save_every,pre_allocate = pre_allocate) 
0
t_test = t[1:end]
test_sim = test_sim[:,:,:,1,:]
W_test_sim = PO.W(test_sim)

# ** Compute predicted data **
t_start = 0
t_end = 5
dt = 0.05
save_every = 1
pre_allocate = true


t,pred_sim = simulate(PO.W(new_init_conds),coarse_mesh,dt,t_start,t_end,neural_rhs,time_step,save_every = save_every,pre_allocate = pre_allocate) 
t_pred = t[1:end]
pred_sim = pred_sim[:,:,:,1,:]

R_pred_sim = PO.R(pred_sim)


plot(t_test,coarse_mesh.ip(W_test_sim,W_test_sim,combine_channels = false)[1,1,1,:],label = "DNS",linewidth = 2,xguidefontsize=14,yguidefontsize=14 ,dpi = 130, legend=:topright,legendfont=font(15))
plot!(t_pred,coarse_mesh.ip(pred_sim,pred_sim,combine_channels = false)[1,1,1,:],label = "SP",linewidth = 2)
ylabel!(L"$\bar{E}_h$")
xlabel!(L"t")


plot(mean(coarse_mesh.ip(W_test_sim,W_test_sim,combine_channels = false),dims = [1,2,4])[1:end],yscale = :log,label = "DNS",marker = true,xguidefontsize=14,yguidefontsize=14 ,dpi = 130, legend=:topright,legendfont=font(15),xscale = :log)
plot!(mean(coarse_mesh.ip(pred_sim,pred_sim,combine_channels = false),dims = [1,2,4])[1:end],label = "SP",marker = true)
xlabel!(L"i")
ylabel!(L"||\mathbf{s}_i||^2_\Omega")



E_ref = fine_mesh.ip(test_sim,test_sim)[1:end]
E_ref = coarse_mesh.ip(PO.W(test_sim),PO.W(test_sim),combine_channels = true)[1:end]
plot(t_test,E_ref,label = "True",linewidth = 2,xguidefontsize=18,yguidefontsize=18 ,dpi = 130, legend=:topright,legendfont=font(15))
plot!(t_pred,E_pred,label = "Simulation",linewidth = 2)
ylabel!(L"$E$")
xlabel!(L"t")


to_plot = PO.R(PO.W(test_sim))
ymin = minimum(to_plot)
ymax = maximum(to_plot)
anim = @animate for index in 1:size(to_plot)[4]
    #heatmap(to_plot[:,:,1,index],color = :brg,clim = (ymin,ymax))
    heatmap(to_plot[:,:,1,index],color = :brg,aspect_ratio = :equal,right_margin =55Plots.mm)
    xlabel!(L"x")
    ylabel!(L"y")
    title!("DNS")
    
end
gif(anim, "DNS_flow.gif", fps = 10)



to_plot = R_pred_sim
ymin = minimum(to_plot)
ymax = maximum(to_plot)
anim = @animate for index in 1:size(to_plot)[4]
    heatmap(to_plot[:,:,1,index],color = :brg,aspect_ratio = :equal,right_margin =35Plots.mm)
    xlabel!(L"x")
    ylabel!(L"y")
    title!("SP")
end
gif(anim, "SP_flow.gif", fps = 10)

