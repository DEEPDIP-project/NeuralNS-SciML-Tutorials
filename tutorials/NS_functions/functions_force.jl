
# The function `Q` computes the quadratic term.
# The `K - Kf` highest frequencies of `u` are cut-off to prevent aliasing.
function Q(u, params)
    n = size(u, 1)
    Kz = params.K - params.Kf

    ## Remove aliasing components
    uf = [
        u[1:params.Kf, 1:params.Kf, :] z(params.Kf, 2Kz, 2) u[1:params.Kf, end-params.Kf+1:end, :]
        z(2Kz, params.Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, params.Kf, 2)
        u[end-params.Kf+1:end, 1:params.Kf, :] z(params.Kf, 2Kz, 2) u[end-params.Kf+1:end, end-params.Kf+1:end, :]
    ]

    ## Spatial velocity
    v = real.(ifft(uf, (1, 2)))
    vx, vy = eachslice(v; dims = 3)

    ## Quadractic terms in space
    vxx = vx .* vx
    vxy = vx .* vy
    vyy = vy .* vy
    v2 = cat(vxx, vxy, vxy, vyy; dims = 3)
    v2 = reshape(v2, n, n, 2, 2)

    ## Quadractic terms in spectral space
    q = fft(v2, (1, 2))
    qx, qy = eachslice(q; dims = 4)

    ## Compute partial derivatives in spectral space
    ∂x = 2.0f0π * im * params.k
    ∂y = 2.0f0π * im * reshape(params.k, 1, :)
    q = @. -∂x * qx - ∂y * qy

    ## Zero out high wave-numbers (is this necessary?)
    q = [
        q[1:params.Kf, 1:params.Kf, :] z(params.Kf, 2Kz, 2) q[1:params.Kf, params.Kf+2Kz+1:end, :]
        z(2Kz, params.Kf, 2) z(2Kz, 2Kz, 2) z(2Kz, params.Kf, 2)
        q[params.Kf+2Kz+1:end, 1:params.Kf, :] z(params.Kf, 2Kz, 2) q[params.Kf+2Kz+1:end, params.Kf+2Kz+1:end, :]
    ]

    q
end

# `F` computes the unprojected momentum right hand side $\hat{F}$. It also
# includes the closure term (if any).
function F(u, params)
    q = Q(u, params)
    du = @. q - params.prefactor_F * u + params.f
    du
end

# The projector $P$ uses pre-assembled matrices.
function project(u, params)
    ux, uy = eachslice(u; dims = 3)
    dux = @. params.Pxx * ux + params.Pxy * uy
    duy = @. params.Pxy * ux + params.Pyy * uy
    cat(dux, duy; dims = 3)
end
