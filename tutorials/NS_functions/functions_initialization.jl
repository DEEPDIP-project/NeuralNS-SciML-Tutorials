
# For the initial conditions, we create a random spectrum with some decay.
# Note that the initial conditions are projected onto the divergence free
# space at the end.

function create_spectrum(params; A, σ, s)
    T = eltype(params.x)
    kx = params.k
    ky = reshape(params.k, 1, :)
    τ = 2.0f0π
    a = @. A / sqrt(τ^2 * 2σ^2) *
       exp(-(kx - s)^2 / 2σ^2 - (ky - s)^2 / 2σ^2 - im * τ * rand(T))
    a
end

function random_field(params; A = 1.0f6, σ = 30.0f0, s = 5.0f0)
    ux = create_spectrum(params; A, σ, s)
    uy = create_spectrum(params; A, σ, s)
    u = cat(ux, uy; dims = 3)
    u = real.(ifft(u, (1, 2)))
    u = fft(u, (1, 2))
    project(u, params)
end