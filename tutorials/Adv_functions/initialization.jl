
function construct_k(N)
    dims = length(N)
    k = [fftfreq(i,i) for i in N]

    some_ones = ones(N)
    k_mats = some_ones .* k[1]

    k_mats = reshape(k_mats,(size(k_mats)...,1))

    for i in 2:dims
        original_dims = collect(1:dims)
        permuted_dims = copy(original_dims)
        permuted_dims[1] = i
        permuted_dims[i] = 1

        k_mat = permutedims(k[i] .* permutedims(some_ones,permuted_dims),permuted_dims)
        k_mats = cat(k_mats,k_mat,dims = dims + 1)
    end
    return k_mats
end


function construct_spectral_filter(k_mats,max_k)
    filter = ones(size(k_mats)[1:end-1])
    N = size(k_mats)[1:end-1]
    dims = length(N)
    loop_over = gen_permutations(N)
    for i in 1:size(loop_over)[1]
        i = loop_over[i,:]
        k = k_mats[i...,:]
        if sqrt(sum(k.^2)) >= max_k
            filter[i...] = 0
        end
    end
    return filter
end

function generate_random_field(N,max_k;norm = 1,samples = (1,1))
    dims = length(N)
    k = construct_k(N)
    filter = construct_spectral_filter(k,max_k)
    coefs = (rand(Uniform(-1,1),(N...,samples...)) + rand(Uniform(-1,1),(N...,samples...)) * (0+1im))

    result = real.(ifft(filter .* coefs,collect(1:dims)))
    sqrt_energies = sqrt.(1/prod(N) .* sum(result.^2,dims = collect(1:dims)))
    result ./= sqrt_energies
    result .*= norm
    return result
end