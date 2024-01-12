
# Random a priori loss function for stochastic gradient descent
# It basically fits the commutator to be equal to the closure
function mean_squared_error(f, st, x, θ; λ = 1.0f-4) 
    batch_size = size(x, 4)
    total_loss = 0.0
    for i in 1:batch_size
        pred_x = Array(f(x[:,:,:,i], θ, st)[1])
        total_loss += sum(abs2, pred_x - x[:,:,:,i]) / sum(abs2, x[:,:,:,i])
    end
    return total_loss + λ * norm(θ), nothing
end
# the loss functions are randomized by selecting a subset of the data, because it would be too expensive to use the entire dataset at each iteration
function create_randloss_derivative(f, st, ubar; nuse = size(ubar, 2))
    d = ndims(ubar)
    nsample = size(ubar, d)
    function randloss(θ)
        i = Zygote.@ignore sort(shuffle(1:nsample)[1:nuse])
        xuse = Zygote.@ignore ArrayType(selectdim(ubar, d, i))
        mean_squared_error(f, st, xuse, θ)
    end
end

# *********************
# DtO Loss for NeuralODE object
function create_randloss_DtO(ubar)
    d = ndims(ubar)
    nt = size(ubar, d)
    function randloss_DtO(p)
        # Zygote will select a random initial condition of lenght nunroll
        istart = Zygote.@ignore rand(1:nt-nunroll)
        trajectory = Zygote.@ignore ArrayType(selectdim(ubar, d, istart:istart+nunroll))
        # this is the loss evaluated for each piece
        loss_DtO_onepiece(trajectory, p)
    end
end
# Piecewise loss function
function loss_DtO_onepiece(trajectory, p)
    tr_start = trajectory[:,:,:,1]
    pred = predict_neuralode(tr_start,p)
    loss = sum(abs2, trajectory .- pred) ./ sum(abs2, trajectory)
    return loss, pred
end
# auxiliary function to solve the NeuralODE, given parameters p
function predict_neuralode(u0,p)
    Array(prob_neuralode(u0, p, st)[1])
end


# *********************
# Multishooting DtO loss for NeuralODE object
function create_randloss_MulDtO(ubar)
    d = ndims(ubar)
    nt = size(ubar, d)
    function randloss_MulDtO(p)
        # Zygote will select a random initial condition that can accomodate all the multishooting intervals
        istart = Zygote.@ignore rand(1:nt-nunroll*nintervals)
        trajectory = Zygote.@ignore ArrayType(selectdim(ubar, d, istart:istart+nunroll*nintervals))
        # this is the loss evaluated for each multishooting set
        loss_MulDtO_oneset(trajectory, p)
    end
end
# the parameter λ sets how strongly we make the pieces match (continuity term)
function loss_MulDtO_oneset(trajectory, p; λ=200)
    loss = 0.0
    last_pred = nothing
    for i in 1:nintervals
        tr_start = trajectory[:,:,:,1+(i-1)*nunroll]
        pred = predict_neuralode(tr_start,p)
        loss += sum(abs2, trajectory[:,:,:,1+(i-1)*nunroll:1+i*nunroll] .- pred) ./ sum(abs2, trajectory[:,:,:,1+(i-1)*nunroll:1+i*nunroll])
        # add continuity term
        if last_pred != nothing
            loss += λ * sum(abs, last_pred .- tr_start) 
        end
        last_pred = pred[:,:,:,end]
    end
    return loss, nothing
end

# *****************
# Testing function to compute the error of a solution vs the DNS
function compute_error(node_solution, v)
    tot_error = 0
    elist = []
    for idx in 1:size(node_solution, 4)
        idx_full = 10*idx
        # if it is the last iteration, set idx_full to the last one
        if idx_full > size(v, 4)
            idx_full = size(v, 4)
        end
        tot_error += sum(abs2, node_solution[:,:,:,idx] - v[:,:,:,idx_full]) / sum(abs2, v[:,:,:,idx_full]) 
        push!(elist, tot_error)
    end
    return elist
end