
# For plotting, the spatial vorticity can be useful. It is given by
#
# $$
# \omega = -\frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x},
# $$
#
# which becomes
#
# $$
# \hat{\omega} = 2 \pi \mathrm{i} k \times u = - 2 \pi \mathrm{i} k_y u_x + 2 \pi \mathrm{i} k_x u_y
# $$
#
# in spectral space.

function vorticity(u, params)
    ∂x = 2f0π * im * params.k
    ∂y = 2f0π * im * reshape(params.k, 1, :)
    ux, uy = eachslice(u; dims = 3)
    ω = @. -∂y * ux + ∂x * uy
    real.(ifft(ω))
end