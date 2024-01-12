
#POD_modes

struct projection_operators_struct
    Phi_T
    Phi
    W
    R
    r
end

function gen_projection_operators(POD_modes,MP;uniform = false)

    dims = MP.fine_mesh.dims
    J = MP.J
    I = MP.I
    sqrt_omega_tilde = sqrt.(MP.omega_tilde)
    inv_sqrt_omega_tilde = 1 ./ sqrt_omega_tilde

    if uniform == false

        Phi_T(input,modes = POD_modes,MP = MP) = cat([sum(MP.one_filter(input .* modes[[(:) for i in 1:MP.fine_mesh.dims+1]...,j]),dims = [MP.fine_mesh.dims+1]) for j in 1:size(modes)[end]]...,dims = MP.fine_mesh.dims + 1)

        function Phi(input,modes = POD_modes,MP = MP)
            UPC = MP.fine_mesh.UPC
            dims = MP.fine_mesh.dims
            r = size(modes)[end]
            Phi_mask = ones((size(input)[1:end-2]...,UPC,size(input)[end]))
            result = stop_gradient() do
                zeros((size(modes)[1:dims]...),UPC,size(input)[end])
            end
            for j in 1:r
                result += modes[[(:) for i in 1:dims+1]...,j:j] .* MP.one_reconstructor(input[[(:) for i in 1:dims]...,j:j,:] .* Phi_mask)
            end
            return result
        end
    else
        weights = POD_modes[[(1:J[i]) for i in 1:dims]...,:,:]

        @assert dims <= 1 "Uniform Phi is not supported for dims > 1 at this time, set uniform = false"

        for i in 1:dims
            weights = reverse(weights,dims = i)
        end

        Phi_T = Conv(J, size(weights)[dims+1]=>size(weights)[dims+2],stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image
        Phi = ConvTranspose(J, size(weights)[dims+2]=>size(weights)[dims+1],stride = J,pad = 0,bias =false)  # First c
        Phi_T.weight .= weights
        Phi.weight .= weights
    end

    W(input,Phi_T = Phi_T, sqrt_omega_tilde = sqrt_omega_tilde) =  Phi_T(input .* sqrt_omega_tilde)
    R(input,Phi = Phi,inv_sqrt_omega_tilde =inv_sqrt_omega_tilde) =  inv_sqrt_omega_tilde .*  Phi(input)

    return projection_operators_struct(Phi_T,Phi,W,R,r)
end