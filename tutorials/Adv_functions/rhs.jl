# These functions focus on the right hand side of the equation, i.e. the function that computes the time derivative of the solution.


# Option 1: Convective advection equation
function gen_conv_advection_rhs(speed,viscosity)
    conv_D1 = gen_conv_stencil([[0, 1, 0];;[1, 0, -1];;[0, -1, 0]])
    conv_D2 = gen_conv_stencil([[0, 1, 0];;[1, -4, 1];;[0, 1, 0]])
    function advection_rhs(u,mesh,t,D1=conv_D1,D2 = conv_D2,viscosity =  viscosity,speed = speed)
        dx = mesh.omega[1,1]
        A = 1/(2*dx) * D1(u)
        B = (1/(dx^2))*D2(u)
        return -speed*A .+ viscosity*B
    end
    return advection_rhs
end
function gen_conv_stencil(weights)
    widths = [(Int.(([size(weights)[1:end]...] .- 1) ./ 2)...,)]
    conv_stencil = conv_NN(widths,[1,1],[(1,1)],false)
    conv_stencil[1].weight[[(:) for i in 1:length(widths[1])]...] = weights#reverse(weights)
    return conv_stencil#conv_stencil
end


# Option 2: Burgers equation
function gen_conv_burgers_rhs(viscosity)
    conv_D1 = gen_conv_stencil([[0, 1, 0];;[1, 0, -1];;[0, -1, 0]])
    conv_D2 = gen_conv_stencil([[0, 1, 0];;[1, -4, 1];;[0, 1, 0]])
    function burgers_rhs(u,mesh,t,D1=conv_D1,D2 = conv_D2,viscosity =  viscosity)
        dx = mesh.omega[1,1]
        A = 1/(2*dx) * 1/3 .* D1(u.^2)
        B = 1/(2*dx) * 1/3 * u[2:end-1,2:end-1,:,:] .* D1(u)
        C = (1/(dx^2))*D2(u)
        return -(A+B) .+ viscosity*C
    end
    return burgers_rhs
end