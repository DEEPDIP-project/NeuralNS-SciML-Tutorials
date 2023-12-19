
# #### Wavelet neural operator architecture
#
# Now let's implement the Wavelet Neural Operator (WNO).

struct WaveletLayer{A,F} <: Lux.AbstractExplicitLayer
    # this will dictate the maximum number of coefficients to include in the DWT
    nwmax::Int
    cin::Int
    cout::Int
    σ::A
    init_weight::F
end

WaveletLayer(nwmax, ch::Pair{Int,Int}; σ = identity, init_weight = glorot_uniform) =
    WaveletLayer(nwmax, first(ch), last(ch), σ, init_weight)


Lux.initialparameters(rng::AbstractRNG, (; nwmax, cin, cout, init_weight)::WaveletLayer) = (;
    spatial_weight = init_weight(rng, cout, cin),
    wavelet_weights = init_weight(rng, nwmax + 1, nwmax + 1, cout, cin, 2),
)
Lux.initialstates(::AbstractRNG, ::WaveletLayer) = (;)
Lux.parameterlength((; kmax, cin, cout)::WaveletLayer) =
    cout * cin + (kmax + 1)^2 * 2 * cout * cin
Lux.statelength(::WaveletLayer) = 0

# We now define how to pass inputs through Wavelet layer, assuming the
# following:
#
# - Input size: `(N, N, cin, nsample)`
# - Output size: `(N, N, cout, nsample)`

function ((; nwmax, cout, cin, σ)::WaveletLayer)(x, params, state)

    ## TBD 
    ## TBD 
    ## TBD 
    #N = size(x, 1)

    ### Destructure params
    ### The real and imaginary parts of R are stored in two separate channels
    #W = params.spatial_weight
    #W = reshape(W, 1, 1, cout, cin)
    #R = params.spectral_weights
    #R = selectdim(R, 5, 1) .+ im .* selectdim(R, 5, 2)

    ### Spatial part (applied point-wise)

    #y = reshape(x, N, N, 1, cin, :)
    #y = sum(W .* y; dims = 4)
    #y = reshape(y, N, N, cout, :)

    ### Spectral part (applied mode-wise)
    ###
    ### Steps:
    ###
    ### - go to complex-valued spectral space
    ### - chop off high wavenumbers
    ### - multiply with weights mode-wise
    ### - pad with zeros to restore original shape
    ### - go back to real valued spatial representation
    #ikeep = (1:kmax+1, 1:kmax+1)
    #nkeep = (kmax + 1, kmax + 1)
    #dims = (1, 2)
    #z = fft(x, dims)
    #z = z[ikeep..., :, :]
    #z = reshape(z, nkeep..., 1, cin, :)
    #z = sum(R .* z; dims = 4)
    #z = reshape(z, nkeep..., cout, :)
    #z = pad_zeros(z, (0, N - kmax - 1, 0, N - kmax - 1); dims)
    #z = real.(ifft(z, dims))

    ### Outer layer: Activation over combined spatial and spectral parts
    ### Note: Even though high wavenumbers are chopped off in `z` and may
    ### possibly not be present in the input at all, `σ` creates new high
    ### wavenumbers. High wavenumber functions may thus be represented using a
    ### sequence of Fourier layers. In this case, the `y`s are the only place
    ### where information contained in high input wavenumbers survive in a
    ### Fourier layer.
    #v = σ.(y .+ z)

    ## Fourier layer does not modify state
    v, state
end