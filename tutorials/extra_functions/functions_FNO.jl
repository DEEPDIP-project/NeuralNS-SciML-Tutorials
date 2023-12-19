
# #### Fourier neural operator architecture
#
# Now let's implement the Fourier Neural Operator (FNO) [^3].
# A Fourier layer $u \mapsto w$ is given by the following expression:
#
# $$
# w(x) = \sigma \left( z(x) + W u(x) \right)
# $$
#
# where $\hat{z}(k) = R(k) \hat{u}(k)$ for some weight matrix collection
# $R(k) \in \mathbb{C}^{n_\text{out} \times n_\text{in}}$. The important
# part is the following choice: $R(k) = 0$ for $\| k \| > k_\text{max}$
# for some $k_\text{max}$. This truncation makes the FNO applicable to
# any discretization as long as $K > k_\text{max}$, and the same parameters
# may be reused.
#
# The deep learning framework [Lux](https://lux.csail.mit.edu/) let's us define
# our own layer types. Everything should be explicit ("functional
# programming"), including random number generation and state modification. The
# weights are stored in a vector outside the layer, while the layer itself
# contains information for construction the network.

struct FourierLayer{A,F} <: Lux.AbstractExplicitLayer
    kmax::Int
    cin::Int
    cout::Int
    σ::A
    init_weight::F
end

FourierLayer(kmax, ch::Pair{Int,Int}; σ = identity, init_weight = glorot_uniform) =
    FourierLayer(kmax, first(ch), last(ch), σ, init_weight)

# We also need to specify how to initialize the parameters and states. The
# Fourier layer does not have any hidden states (RNGs) that are modified.

Lux.initialparameters(rng::AbstractRNG, (; kmax, cin, cout, init_weight)::FourierLayer) = (;
    spatial_weight = init_weight(rng, cout, cin),
    spectral_weights = init_weight(rng, kmax + 1, kmax + 1, cout, cin, 2),
)
Lux.initialstates(::AbstractRNG, ::FourierLayer) = (;)
Lux.parameterlength((; kmax, cin, cout)::FourierLayer) =
    cout * cin + (kmax + 1)^2 * 2 * cout * cin
Lux.statelength(::FourierLayer) = 0

# We now define how to pass inputs through Fourier layer, assuming the
# following:
#
# - Input size: `(N, N, cin, nsample)`
# - Output size: `(N, N, cout, nsample)`

function ((; kmax, cout, cin, σ)::FourierLayer)(x, params, state)
    N = size(x, 1)

    ## Destructure params
    ## The real and imaginary parts of R are stored in two separate channels
    W = params.spatial_weight
    W = reshape(W, 1, 1, cout, cin)
    R = params.spectral_weights
    R = selectdim(R, 5, 1) .+ im .* selectdim(R, 5, 2)

    ## Spatial part (applied point-wise)

    y = reshape(x, N, N, 1, cin, :)
    y = sum(W .* y; dims = 4)
    y = reshape(y, N, N, cout, :)

    ## Spectral part (applied mode-wise)
    ##
    ## Steps:
    ##
    ## - go to complex-valued spectral space
    ## - chop off high wavenumbers
    ## - multiply with weights mode-wise
    ## - pad with zeros to restore original shape
    ## - go back to real valued spatial representation
    ikeep = (1:kmax+1, 1:kmax+1)
    nkeep = (kmax + 1, kmax + 1)
    dims = (1, 2)
    z = fft(x, dims)
    z = z[ikeep..., :, :]
    z = reshape(z, nkeep..., 1, cin, :)
    z = sum(R .* z; dims = 4)
    z = reshape(z, nkeep..., cout, :)
    z = pad_zeros(z, (0, N - kmax - 1, 0, N - kmax - 1); dims)
    z = real.(ifft(z, dims))

    ## Outer layer: Activation over combined spatial and spectral parts
    ## Note: Even though high wavenumbers are chopped off in `z` and may
    ## possibly not be present in the input at all, `σ` creates new high
    ## wavenumbers. High wavenumber functions may thus be represented using a
    ## sequence of Fourier layers. In this case, the `y`s are the only place
    ## where information contained in high input wavenumbers survive in a
    ## Fourier layer.
    v = σ.(y .+ z)

    ## Fourier layer does not modify state
    v, state
end

# Function to create the model
function create_fno_model(kmax_fno, ch_fno, σ_fno; single_timestep = false)
    return Chain(
        u -> real.(ifft(u, (1, 2))),
        # Raise an error if u has less than 4 dimensions
        (
            FourierLayer(kmax_fno[i], ch_fno[i] => ch_fno[i+1]; σ = σ_fno[i]) for
            i ∈ eachindex(σ_fno)
        )...,
        u -> permutedims(u, (3, 1, 2, 4)),
        Dense(ch_fno[end] => 2 * ch_fno[end], gelu),
        Dense(2 * ch_fno[end] => 2; use_bias = false),
        u -> permutedims(u, (2, 3, 1, 4)),
        u -> fft(u, (1, 2)),
        # If I am using the layer for single time step, I need to reshape
        u -> single_timestep ? dropdims(u,dims=4) : u,
    )
end


# Function to generate the model name
function generate_FNO_name(ch_fno, kmax_fno, σ_fno)
    ch_str = join(ch_fno, '-')
    kmax_str = join(kmax_fno, '-')
    σ_str = join(σ_fno, '-')
    
    return "FNO__$(ch_str)__$(kmax_str)__$(σ_str)"
end