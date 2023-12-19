mutable struct Params
    x
    N
    K
    Kf
    k
    nu
    normk
    f
    Pxx
    Pxy
    Pyy
    prefactor_F
end

# Store parameters and precomputed operators in a named tuple to toss around.
# Having this in a function gets useful when we later work with multiple
# resolutions.
function create_params(
    K;
    nu,
    f = z(2K, 2K),
    anti_alias_factor = 2 / 3,
)
    Kf = round(Int, anti_alias_factor * K)
    N = 2K
    x = LinRange(0.0f0, 1.0f0, N + 1)[2:end]

    ## Vector of wavenumbers
    k = ArrayType(fftfreq(N, Float32(N)))
    normk = k .^ 2 .+ k' .^ 2

    ## Projection components
    kx = k
    ky = reshape(k, 1, :)
    Pxx = @. 1 - kx * kx / (kx^2 + ky^2)
    Pxy = @. 0 - kx * ky / (kx^2 + ky^2)
    Pyy = @. 1 - ky * ky / (kx^2 + ky^2)

    ## The zeroth component is currently `0/0 = NaN`. For `CuArray`s,
    ## we need to explicitly allow scalar indexing.

    CUDA.@allowscalar Pxx[1, 1] = 1
    CUDA.@allowscalar Pxy[1, 1] = 0
    CUDA.@allowscalar Pyy[1, 1] = 1

    # Prefactor for F
    pf = Zygote.@ignore nu * (2.0f0π)^2 * normk

    Params(x, N, K, Kf, k, nu, normk, f, Pxx, Pxy, Pyy, pf)
end
