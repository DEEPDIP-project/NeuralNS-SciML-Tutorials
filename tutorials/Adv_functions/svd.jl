# Functions to perform Single Value Decomposition (SVD)

function gen_permutations(N)

    N_grid = [collect(1:n) for n in N]
    sub_grid = ones(Int,(N...))
    dims = length(N)
    sub_grids = []

    for i in 1:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = original_dims[i]
        permuted_dims[i] = 1

        push!(sub_grids,permutedims(N_grid[i] .*  permutedims(sub_grid,permuted_dims),permuted_dims))
    end

    return reshape(cat(sub_grids...,dims = dims + 1),(prod(N)...,dims))
end


function reshape_for_local_SVD(input,MP; subtract_average = false)
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims
    dims = length(J)

    offsetter = [J...]
    loop_over = gen_permutations(I)
    data = []

    for i in 1:size(loop_over)[1]
        i = loop_over[i,:]
        first_index = offsetter .* (i .-1 ) .+ 1
        second_index = offsetter .* (i)
        index = [(first_index[i]:second_index[i]) for i in 1:dims]
        index = [index...,(:),(:)]
        to_push = input[index...]
        if subtract_average
            to_push .-= mean(to_push,dims = collect(1:dims))
        end
        push!(data,to_push)
    end

    return cat(data...,dims = dims + 2)
end


function carry_out_local_SVD(input,MP;subtract_average = false)
    UPC = MP.coarse_mesh.UPC
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims
    reshaped_input = reshape_for_local_SVD(input,MP,subtract_average = subtract_average)

    vector_input = reshape(reshaped_input,(prod(size(reshaped_input)[1:end-1]),size(reshaped_input)[end]))

    SVD = svd(vector_input)
    return reshape(SVD.U,(J...,UPC,Int(size(SVD.U)[end]))),SVD.S
end


# Transform a set of modes from local representation to global
function local_to_global_modes(modes,MP)
    number_of_modes = size(modes)[end]
    UPC = MP.coarse_mesh.UPC
    J = MP.J
    I = MP.I
    dims = MP.fine_mesh.dims

    some_ones = ones(size(modes)[1:end]...,prod(I))
    global_modes = modes .* some_ones

    original_dims = collect(1:length(size(global_modes)))
    permuted_dims = copy(original_dims)
    permuted_dims[end] = original_dims[end-1]
    permuted_dims[end-1] = original_dims[end]

    global_modes = permutedims(global_modes,permuted_dims)
    global_modes = reshape(global_modes,(J...,UPC, I...,number_of_modes))
    output = zeros(I..., J...,UPC,number_of_modes)

    loop_over = gen_permutations((J...,UPC))
    for i in 1:size(loop_over)[1]
        i = loop_over[i,:]
        output[[(:) for j in 1:dims]...,i...,:] = global_modes[i...,[(:) for j in 1:dims]...,:]
    end

    to_reconstruct = reshape(output,(I..., prod(J)*UPC,number_of_modes))
    return reshape(reconstruct_signal(to_reconstruct,J),(([I...] .* [J...])...,UPC,number_of_modes))
end