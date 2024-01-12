
function trajectory_fitting_loss(input,indexes,times,dt = traj_dt,steps = traj_steps,sim_interpolator = sim_interpolator,neural_rhs = neural_rhs,coarse_mesh = coarse_mesh)
    dims = length(size(input)) - 2
    
    t_start = reshape(times,([1 for i in 1:dims+1]...,size(times)[1]))
    t_end =  t_start .+ steps *dt
    t,result = simulate(input,coarse_mesh,dt,t_start,t_end,neural_rhs,time_step,save_every = 1,pre_allocate = false) 
    reference = stop_gradient() do
        sim_interpolator(t,simulation_indexes = indexes)
    end
    return Flux.Losses.mse(result,reference)
end    


function derivative_fitting_loss(a,rhs_a,neural_rhs =neural_rhs,coarse_mesh = coarse_mesh)
    pred = neural_rhs(a,coarse_mesh,0)
    return Flux.Losses.mse(pred,rhs_a) #+ Flux.Losses.mse(coarse_mesh.ip(pred,a),coarse_mesh.ip(rhs_a,a))
end