
function compute_overlap_matrix(modes)
    dims = length(size(modes)) -2
    overlap = zeros(size(modes)[end],size(modes)[end])
    for i in 1:size(modes)[end]
        input_1 = modes[[(:) for k in 1:dims+1]...,i:i]
        for j in 1:size(modes)[end]
            input_2 = modes[[(:) for k in 1:dims+1]...,j:j]
            overlap[i,j] = sum(input_1 .* input_2, dims = collect(1:dims+1))[1]
        end
    end
    return overlap
end

function add_filter_to_modes(POD_modes,MP;orthogonalize = false)

    dims = MP.fine_mesh.dims
    UPC = MP.fine_mesh.UPC
    sqrt_omega_tilde = sqrt.(MP.omega_tilde)
    some_zeros = zeros(size(MP.omega_tilde))

    modes = cat([sqrt_omega_tilde,(some_zeros for i in 1:UPC-1)...]...,dims = dims + 1)
    modes = cat([circshift(modes,([0 for i in 1:dims]...,j)) for j in 0:(UPC-1)]...,dims = dims + 2)
    if POD_modes != 0
        modes = cat([modes,POD_modes]...,dims = dims + 2)
    end

    r = size(modes)[dims + 2]
    IP = 0
    for i in 2:r
        s_i = [[(:) for k in 1:dims+1]...,i:i]
        mode_i = modes[s_i...]
        if orthogonalize ### orthogonalize basis using gramm-schmidt
            for j in 1:(i-1)
                s_j = [[(:) for k in 1:dims+1]...,j:j]
                mode_j = modes[s_j...]
                IP = sum(MP.one_reconstructor(1/(prod(MP.fine_mesh.N))*MP.one_filter(mode_j .* mode_i)),dims = collect(1:dims+1))
                modes[s_i...] .-= (IP) .* mode_j
            end
            mode_i = modes[s_i...]
        end
        norm_i =  sum(MP.one_reconstructor(1/(prod(MP.fine_mesh.N))*MP.one_filter(mode_i .* mode_i)),dims = collect(1:dims+1))
        modes[s_i...] ./= sqrt.(norm_i)
    end

    return modes
end