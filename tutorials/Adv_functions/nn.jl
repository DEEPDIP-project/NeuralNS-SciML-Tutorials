
###################### Neural network code ############

# What to do with padding
# What to do with multiple unknowns, i.e. u and v field
# How to save the neural network

struct model_struct
    eval
    CNN
    r
    UPC
    pad_size
    boundary_padding
    constrain_energy
    conserve_momentum
    dissipation
    kernel_sizes
    channels
    strides
end

############### Code for skew symmetric neural network #######################


function cons_mom_B(B_kernel;channel = 1)
    if B_kernel != 0
        dims = length(size(B_kernel))-2
        channel_mask = gen_channel_mask(B_kernel,channel)

        means = mean(B_kernel,dims = collect(1:dims))
        return B_kernel .- means .* channel_mask
    else
        return 0
    end
end

function transpose_B(B_kernel)
    if B_kernel != 0
        dims = length(size(B_kernel))-2
        original_dims = stop_gradient() do
           collect(1:dims+2)
        end
        permuted_dims = stop_gradient() do
           copy(original_dims)
        end

        stop_gradient() do
            permuted_dims[dims+1] = original_dims[dims+2]
            permuted_dims[dims+2] = original_dims[dims+1]
        end

        T_B_kernel = permutedims(B_kernel,permuted_dims)

        for i in 1:dims
           T_B_kernel = reverse(T_B_kernel,dims = i)

        end

        return T_B_kernel
    else
        return 0
    end
end


struct skew_model_struct
    eval
    CNN
    r
    B
    B_mats
    UPC
    pad_size
    boundary_padding
    constrain_energy
    conserve_momentum
    dissipation
    kernel_sizes
    channels
    strides
end


function gen_skew_NN(kernel_sizes,channels,strides,r,B;UPC = 0,boundary_padding = 0,constrain_energy = true,conserve_momentum=true,dissipation = true)
    # Adjust the input channels to account for the boundary padding
    if boundary_padding != 0 && boundary_padding != "c"
        add_input_channel = zeros(Int,size(channels)[1]+1)
        add_input_channel[1] += 1
    else
        add_input_channel = 0
    end

    if dissipation && constrain_energy
       channels = [channels ; 2*r]
    else
       channels = [channels ; r]
    end

    # Construct a CNN with the given parameters
    CNN = conv_NN(kernel_sizes,channels .+ add_input_channel,strides)
    pad_size = find_padding_size(CNN)

    if UPC == 0
        UPC = length(size(model.CNN[1].weight))-2
    end
    dims = length(size(CNN[1].weight))-2

    # Construct the B matrices, which are necessary for the energy conservation in the skew-symmetric form
    B1,B2,B3 = 0,0,0
    if constrain_energy
        B1 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r))
        B2 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r))
        if dissipation
            B3 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,r,r))
        end
    end
    # Adjust the padding size to account for the energy conservation
    if constrain_energy
        pad_size = [pad_size...]
        pad_size .+= [B...]
        pad_size = Tuple(pad_size)
    end
    B_mats = [B1,B2,B3]

    # This nested function is the actual neural network that gets applied to the input
    function NN(input;a = 0,CNN = CNN,r = r,B = [B...],B_mats = B_mats,UPC = UPC,pad_size = pad_size,boundary_padding = boundary_padding,constrain_energy =constrain_energy,conserve_momentum = conserve_momentum,dissipation = dissipation)

        dims = length(size(input)) - 2

        if boundary_padding == 0 || boundary_padding == "c"
            output = CNN(padding(input,pad_size,circular = true))

        else
            pad_input = padding(input,pad_size,BCs = boundary_padding)
            boundary_indicator_channel = stop_gradient() do
                ones(size(input)[1:end-2]...,1,size(input)[end])
            end
            boundary_indicator_padding = stop_gradient() do
                copy(boundary_padding)
            end
            stop_gradient() do
                for i in 1:prod(size(boundary_indicator_padding))
                    if boundary_indicator_padding[i] != "c"
                        boundary_indicator_padding[i] = i + 1
                    end
                end
            end
            pad_boundary_indicator_channel = padding(boundary_indicator_channel,pad_size,BCs = boundary_indicator_padding)
            output = CNN(cat([pad_input,pad_boundary_indicator_channel]...,dims = dims + 1))
        end

        phi = output[[(:) for i in 1:dims]...,1:r,:]
        psi = 0
        if constrain_energy && dissipation
            psi = output[[(:) for i in 1:dims]...,r+1:2*r,:]
            #dTd = sum( d.^2 ,dims = [i for i in 1:dims])
        else
            psi = 0
        end

        # Adjust the B matrices to account for the momentum conservation
        B1,B2,B3 = B_mats
        if conserve_momentum && constrain_energy
            B1 = cons_mom_B(B1)
            B2 = cons_mom_B(B2)
            B3 = cons_mom_B(B3)
        end
        B1_T,B2_T,B3_T = 0,0,0
        if constrain_energy
            B1_T = transpose_B(B1)
            B2_T = transpose_B(B2)
            B3_T = transpose_B(B3)
        else
            B1_T,B2_T,B3_T = 0,0,0
        end

        # Construct the skew-symmetric form
        c_tilde = 0
        if constrain_energy 
            c_tilde = NNlib.conv(NNlib.conv(a,B1) .* phi,B2_T) - NNlib.conv(NNlib.conv(a,B2) .* phi,B1_T)
            if dissipation
                c_tilde -=  NNlib.conv(psi.^2 .* NNlib.conv(a,B3),B3_T)
            end
        else
            c_tilde = phi
        end

        return  c_tilde
    end

    return skew_model_struct(NN,CNN,r,B,B_mats,UPC,pad_size,boundary_padding,constrain_energy,conserve_momentum,dissipation,kernel_sizes,channels,strides)
end


function save_skew_model(model,name)
    if name[end] == "/"
        name = name[1:end-1]
    end
    mkpath(name)
    save(name * "/model_state.jld","CNN_weights_and_biases",[(i.weight,i.bias) for i in model.CNN],"r",model.r,"B",model.B,"B_mats",model.B_mats,"UPC",model.UPC,"pad_size",model.pad_size,"boundary_padding",model.boundary_padding,"constrain_energy",model.constrain_energy,"conserve_momentum",model.conserve_momentum,"dissipation",model.dissipation,"kernel_sizes",model.kernel_sizes,"channels",model.channels,"strides",model.strides)
    print("\nModel saved at directory [" * name * "]\n")
end

function load_skew_model(name)
    if name[end] == "/"
        name = name[1:end-1]
    end
    CNN_weights_and_biases,r,B,B_mats,UPC,pad_size,boundary_padding,constrain_energy,conserve_momentum,dissipation,kernel_sizes,channels,strides = (load(name * "/model_state.jld")[i] for i in ("CNN_weights_and_biases","r","B","B_mats","UPC","pad_size","boundary_padding","constrain_energy","conserve_momentum","dissipation","kernel_sizes","channels","strides"))

    model = gen_skew_NN(kernel_sizes,channels,strides,r,B,boundary_padding = boundary_padding,UPC = coarse_mesh.UPC,constrain_energy = constrain_energy,dissipation = dissipation,conserve_momentum = conserve_momentum)

    for i in 1:length(model.CNN)
        model.CNN[i].weight .= CNN_weights_and_biases[i][1]
        model.CNN[i].bias .= CNN_weights_and_biases[i][2]
    end
    for i in 1:length(model.B_mats)
        model.B_mats[i] = B_mats[i]
    end

    print("\nModel loaded from directory [" * name * "]\n")
    return model
end


# Use the NN model to calculate the rhs of the PDE 
function neural_rhs(data,coarse_mesh,t,rhs = rhs,model = model,B=B;other_arguments = 0)
    dims = coarse_mesh.dims
    coarse_rhs = rhs(data[[(:) for i in 1:dims]...,1:1,:],coarse_mesh,t)
    input = cat(data,coarse_rhs,dims = dims + 1)
    
    nn_output = model.eval(input,data = padding(data,((2*[B...])...,),circular = true)) #+ channel_mask .* coarse_rhs
    
    channel_mask = gen_channel_mask(nn_output,1)

    return (1 ./ coarse_mesh.omega) .* nn_output +  channel_mask .* coarse_rhs    
end
