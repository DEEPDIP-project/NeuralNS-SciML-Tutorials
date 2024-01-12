


# figure out how to deal with BCs
# Possibly implement energy conserving auto-encoder
# finish boundary condition indicator (maybe filtered level, maybe ROM level, probably start with filtered)

# connect to NS code
function gen_one_filter(J,UPC)

    #Jx = Int(grid.nx/grid_bar.nx)
    #Jy = Int(grid.ny/grid_bar.ny)
    dims = length(J)
    #J = (Jy,Jx)
    filter = Flux.Conv(J, UPC=>UPC,stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image

    filter.weight .= 1.
    return filter
end


function gen_one_reconstructor(J,UPC)

    #Jx = Int(grid.nx/grid_bar.nx)
    #Jy = Int(grid.ny/grid_bar.ny)
    dims = length(J)
    #J = (Jy,Jx)
    reconstructor = Flux.ConvTranspose(J, UPC=>UPC,stride = J,pad = 0,bias =false)  # First convolution, operating upon a 28x28 image

    reconstructor.weight .= 1.
    return reconstructor
end


function reconstruct_signal(R_q,J)

    ndims(R_q) > 2 || throw(ArgumentError("expected x with at least 3 dimensions"))
    d = ndims(R_q) - 2
    sizein = size(R_q)[1:d]
    cin, n = size(R_q, d+1), size(R_q, d+2)
    #cin % r^d == 0 || throw(ArgumentError("expected channel dimension to be divisible by r^d = $(
    #    r^d), where d=$d is the number of spatial dimensions. Given r=$r, input size(x) = $(size(x))"))

    cout = cin ÷ prod(J)
    R_q = reshape(R_q, sizein..., J..., cout, n)
    perm = hcat(d+1:2d, 1:d) |> transpose |> vec  # = [d+1, 1, d+2, 2, ..., 2d, d]
    R_q = permutedims(R_q, (perm..., 2d+1, 2d+2))
    R_q = reshape(R_q, J.*sizein..., cout, n)

    return R_q
end



function padding(data,pad_size;circular = false,UPC = 0,BCs = 0,zero_corners = true,navier_stokes = false)
    dims = length(size(data)) - 2
    if navier_stokes == false
        UPC = 0
    elseif UPC == 0
        UPC = dims
    end

    if navier_stokes
        if length(size(BCs)) == 0
            BCs = stop_gradient() do
                BCs = BCs*ones(2,dims,UPC)
            end
        end
    else
        if length(size(BCs)) == 0
            BCs = stop_gradient() do
                BCs = BCs*ones(2,dims)
            end
        end
    end

    N = size(data)[1:dims]
    if navier_stokes && (circular == false)
        split_data = [data[[(:) for i in 1:dims]...,j:j,:] for j in 1:UPC]
        unknown_index = 0
        padded_data = []
        for data in split_data
            unknown_index += 1
            for i in 1:dims
                original_dims = stop_gradient() do
                    collect(1:length(size(data)))
                end
                new_dims = stop_gradient() do
                    copy(original_dims)
                end
                stop_gradient() do
                        new_dims[1] = original_dims[i]
                        new_dims[i] = 1
                end

                data = permutedims(data,new_dims)

                pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
                pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]


                if circular == false
                    #@assert one_hot[i][1] != 0 && one_hot[i][2] != 0 "A one-hot encoding of 0 is saved for the corners outside the domain. Use a different number."
                    if BCs[1,i,unknown_index] != "c" && BCs[1,i,unknown_index] != "m"
                        pad_start_cache = 2* BCs[1,i,unknown_index] .- reverse(pad_end,dims = 1)

                    elseif BCs[1,i,unknown_index] == "m"
                        pad_start_cache = reverse(pad_end,dims = 1)
                    else
                        pad_start_cache = pad_start
                    end

                    if BCs[2,i,unknown_index] != "c" && BCs[2,i,unknown_index] != "m"
                        pad_end_cache = 2* BCs[2,i,unknown_index] .- reverse(pad_start,dims = 1)
                    elseif BCs[2,i,unknown_index] == "m"
                        pad_end_cache = reverse(pad_start,dims = 1)
                    else
                        pad_end_cache = pad_end
                    end
                    pad_start = pad_start_cache
                    pad_end = pad_end_cache
                end
                data = cat([pad_start,data,pad_end]...,dims = 1)
                data = permutedims(data,new_dims)
            end
            push!(padded_data,data)
        end

        padded_data = cat(padded_data...,dims = dims + 1)

        if size(data)[dims+1] > UPC
            data = data[[(:) for i in 1:dims]...,UPC+1:end,:]
            for i in 1:dims
                original_dims = stop_gradient() do
                    collect(1:length(size(data)))
                end
                new_dims = stop_gradient() do
                    copy(original_dims)
                end
                stop_gradient() do
                        new_dims[1] = original_dims[i]
                        new_dims[i] = 1
                end
                data = permutedims(data,new_dims)
                pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
                pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]
                if circular == false
                    BC_right = BCs[1,i,:]
                    BC_left = BCs[2,i,:]
                    if BC_right[1] != "c"
                        for j in BC_right
                            @assert j != "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        for j in BC_left
                            @assert j != "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        BC_right = 2*(BC_right .== "m") .- 1
                        BC_left = 2*(BC_left .== "m") .- 1

                        pad_start_cache = reverse(pad_end,dims = 1)
                        pad_end_cache = reverse(pad_start,dims = 1)
                        pad_start = pad_start_cache
                        pad_end = pad_end_cache
                    else
                        for j in BC_right
                            @assert j == "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                        for j in BC_left
                            @assert j == "c" "Mixing periodic BCs with other BC types along dimension " * string(i) * " is not supported"
                        end
                    end
                    pad_start
                end
                data = cat([pad_start,data,pad_end]...,dims = 1)
                data = permutedims(data,new_dims)
            end
            padded_data = cat(padded_data,data,dims = dims + 1)
        end
    else
        for i in 1:dims
            original_dims = stop_gradient() do
                collect(1:length(size(data)))
            end
            new_dims = stop_gradient() do
                copy(original_dims)
            end
            stop_gradient() do
                    new_dims[1] = original_dims[i]
                    new_dims[i] = 1
            end
            data = permutedims(data,new_dims)
            pad_start = data[(end-pad_size[i]+1):end,[(:) for j in 1:(dims+1)]...]
            pad_end = data[1:pad_size[i],[(:) for j in 1:(dims+1)]...]
            if circular == false
                if BCs[1,i,1] != "c" && BCs[1,i,1] != "m"
                    pad_start_cache = BCs[1,i,1] .* pad_start.^0
                elseif BCs[1,i,1] == "m"
                    pad_start_cache = reverse(pad_end,dims = 1)
                else
                    pad_start_cache = pad_start
                end
                if BCs[2,i,1] != "c" && BCs[2,i,1] != "m"
                    pad_end_cache = BCs[2,i,1] .* pad_end.^0
                elseif BCs[2,i,1] == "m"
                    pad_end_cache = reverse(pad_start,dims = 1)
                else
                    pad_end_cache = pad_end
                end
                pad_start = pad_start_cache
                pad_end = pad_end_cache
            end
            data = cat([pad_start,data,pad_end]...,dims = 1)
            data = permutedims(data,new_dims)
        end
        padded_data = data
    end

    if zero_corners == true && circular == false
        corner_mask = stop_gradient() do
            construct_corner_mask(N,pad_size)
        end
        padded_data = padded_data .- corner_mask .* padded_data
    end

    return  padded_data
end










