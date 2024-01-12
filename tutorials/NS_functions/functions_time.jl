
# ## Time discretization
#
# For time stepping, we do a simple fourth order explicit Runge-Kutta scheme.
#
# From a current state $\hat{u}^0 = \hat{u}(t)$, we divide the outer time step
# $\Delta t$ into $s = 4$ sub-steps as follows:
#
# $$
# \begin{split}
# \hat{F}^i & = P \hat{F}(\hat{u}^{i - 1}) \\
# \hat{u}^i & = u^0 + \Delta t \sum_{j = 1}^{i} a_{i j} F^j.
# \end{split}
# $$
#
# The solution at the next outer time step $t + \Delta t$ is then
# $\hat{u}^s = \hat{u}(t + \Delta t) + \mathcal{O}(\Delta t^5)$. The coefficients
# of the RK method are chosen as
#
# $$
# a = \begin{pmatrix}
#     \frac{1}{2} & 0           & 0           & 0 \\
#     0           & \frac{1}{2} & 0           & 0 \\
#     0           & 0           & 1           & 0 \\
#     \frac{1}{6} & \frac{2}{6} & \frac{2}{6} & \frac{1}{6}
# \end{pmatrix}.
# $$
#
# Note that each of the intermediate steps is divergence free.
#
# The following function performs one RK4 time step. Note that we never
# modify any vectors, only create new ones. The AD-framework Zygote prefers
# it this way.

function step_rk4(u0, params, dt)
    a = (
        (0.5f0,),
        (0.0f0, 0.5f0),
        (0.0f0, 0.0f0, 1.0f0),
        (1.0f0 / 6.0f0, 2.0f0 / 6.0f0, 2.0f0 / 6.0f0, 1.0f0 / 6.0f0),
    )
    u = u0
    k = ()
    for i = 1:length(a)
        ki = project(F(u, params), params)
        k = (k..., ki)
        u = u0
        for j = 1:i
            u += dt * a[i][j] * k[j]
        end
    end
    u
end