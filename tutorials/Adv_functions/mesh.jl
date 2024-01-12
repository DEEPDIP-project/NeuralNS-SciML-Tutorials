# Definition of the mesh structure

struct mesh_struct
    dims # 1D/2D
    N # grid resoluiont
    x # coordinates
    x_edges # edges
    omega # mass matrix
    eval_function # evaluate function on the grid
    ip # computes inner-product
    integ # integral on the grid
    UPC # unknows per grid cell
end
# And the structure for the mesh pair (fine and coarse)
struct mesh_pair_struct
    fine_mesh
    coarse_mesh
    J
    I
    one_filter
    one_reconstructor
    omega_tilde
end


# Generates the mesh structure
function gen_mesh(x,y = nothing, z = nothing;UPC=1)
    # If y and z are not nothing, rearrange x, y, z in a certain order
    if y != nothing
        if z != nothing
            x = [z,y,x]
        else
            x = [y,x]
        end
    else
        # If y is nothing and the length of the size of the first element of x is less than or equal to 0, wrap x in an array
        if length(size(x[1])) <= 0
            x = [x]
        end
    end

    # Calculate the midpoints and differences for each pair of consecutive elements in x
    mid_x = [ [(i[j] + i[j+1])/2 for j in 1:(size(i)[1]-1)] for i in x]
    dx = [ [(i[j+1] - i[j]) for j in 1:(size(i)[1]-1)] for i in x]

    # Initialize sub_grid and omega with ones
    sub_grid = ones([size(i)[1] for i in mid_x]...)
    omega = ones([size(i)[1] for i in mid_x]...)

    sub_grids = []
    dims = size(x)[1]

    # Loop over the dimensions
    for i in 1:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = original_dims[i]
        permuted_dims[i] = 1

        # Permute the dimensions of omega and sub_grid
        omega = permutedims(dx[i] .*  permutedims(omega,permuted_dims),permuted_dims)
        push!(sub_grids,permutedims(mid_x[i] .*  permutedims(sub_grid,permuted_dims),permuted_dims))
    end
    x_edges = x
    x = cat(sub_grids...,dims = dims + 1)

    # Define eval_function that evaluates a function F on x
    function eval_function(F,x = x,dims = dims)
        return F([x[[(:) for j in 1:dims]...,i] for i in 1:dims])
    end

    # Define ip (inner product) function
    function ip(a,b;weighted = true,omega = omega,dims = dims,combine_channels = true)
        if weighted
            IP = a .* omega .* b
        else
            IP = a .* b
        end
        if combine_channels == true
            IP =  sum(IP,dims = collect(1:(dims+1)))
        else
            IP =  sum(IP,dims = collect(1:(dims)))
        end
        return IP
    end

    # Define integ (integration) function
    function integ(a;weighted = true,omega = omega,dims = dims,ip = ip)
        some_ones = ones(size(a))
        return ip(some_ones,a,weighted=weighted,omega=omega,dims=dims,combine_channels = false)
    end

    # Return a mesh_struct
    return mesh_struct(dims,size(omega),x,x_edges,omega,eval_function,ip,integ,UPC)
end

# Generates the mesh pair structure from a fine mesh and a reduction parameter J
function generate_coarse_from_fine_mesh(fine_mesh,J)
    divide = [fine_mesh.N...] .% [J...]
    for i in divide
        @assert i == 0 "Meshes are not compatible. Make sure the dimensions of the fine mesh are
                divisible reduction parameter J in each dimension."
    end

    dims = fine_mesh.dims
    N = fine_mesh.N
    x = fine_mesh.x_edges
    I =Tuple([Int(fine_mesh.N[i]/J[i]) for i in 1:dims])

    X  = []
    for i in 1:length(x)
        # select one element every J[i]
        selector = [1,(1 .+ J[i]*collect(1:I[i]))...]
        push!(X,x[i][selector])
    end
    # the coarse mesh has the same unknowns per grid cell as the fine mesh
    return gen_mesh(X,UPC = fine_mesh.UPC)
end

# Generate a mesh pair structure from a fine mesh and a coarse mesh 
function gen_mesh_pair(fine_mesh,coarse_mesh)
    divide = [fine_mesh.N...] .% [coarse_mesh.N...]
    for i in divide
        @assert i == 0 "Meshes are not compatible. Make sure the dimensions of the fine mesh are
                divisible by the dimensions of the coarse mesh."
    end
    UPC = fine_mesh.UPC
    dims = fine_mesh.dims
    J =Tuple([Int(fine_mesh.N[i]/coarse_mesh.N[i]) for i in 1:dims])
    I = coarse_mesh.N

    one_filter = gen_one_filter(J,UPC)
    one_reconstructor = gen_one_reconstructor(J,UPC)
    omega_tilde = fine_mesh.omega

    omega_UPC = cat([coarse_mesh.omega for i in 1:UPC]...,dims = dims + 1)
    omega_UPC = reshape(omega_UPC,(size(omega_UPC))...,1)
    omega_tilde = fine_mesh.omega ./ one_reconstructor(omega_UPC)[[(:) for i in 1:dims]...,1,1]

    return mesh_pair_struct(fine_mesh,coarse_mesh,J,I,one_filter,one_reconstructor,omega_tilde)
end
