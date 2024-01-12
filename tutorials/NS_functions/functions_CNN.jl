
# #### Convolutional neural network
#
function create_cnn_model(r_cnn, ch_cnn, σ_cnn, b_cnn; single_timestep = false)
    return Chain(
        ## Go to physical space
        u -> real.(ifft(u, (1, 2))),
        ## Add padding so that output has same shape as commutator error
        u -> single_timestep ? reshape(u, (size(u)...,1)) : u,
        u -> pad_circular(u, sum(r_cnn)),
        ## Some convolutional layers
        (
            Conv(
                (2 * r_cnn[i] + 1, 2 * r_cnn[i] + 1),
                ch_cnn[i] => ch_cnn[i+1],
                σ_cnn[i];
                use_bias = b_cnn[i],
            ) for i ∈ eachindex(r_cnn)
        )...,
        ## Go to spectral space
        u -> fft(u, (1, 2)),
        # If I am using the layer for single time step, I need to reshape
        u -> single_timestep ? dropdims(u,dims=4) : u,
    )
end


# Function to generate the model name
function generate_CNN_name(r_cnn, ch_cnn, σ_fno, b_cnn)
    r_str = join(r_cnn, '-')
    ch_str = join(ch_cnn, '-')
    σ_str = join(σ_cnn, '-')
    b_str = join(b_cnn, '-')
    
    return "CNN__$(r_str)__$(ch_str)__$(σ_str)__$(b_str)"
end