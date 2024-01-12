# I define the NeuralODE using ResNet skip blocks
# the prefactor_closure is mainly used to remove the colosure if desired
function create_node(closure,simulation_params; is_closed=false)
    return Chain(
        SkipConnection(closure, (closureu, u) -> is_closed ? closureu + F(u, simulation_params) : F(u, simulation_params)),
        u -> project(u, simulation_params),
    )
end


# This is a test colosure model
function create_test_closure(n,l)
    return Chain(
        ## Go to physical space
        u -> real.(FFTW.ifft(u, (1, 2))),
        u -> vcat(u...),
        Dense(n => 1),
        Dense(1 => n),
        u -> reshape(u,(l,l,2)),
        # back to Fourier space
        u -> FFTW.fft(u, (1, 2)),
    )
end