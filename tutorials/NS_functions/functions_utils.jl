
# We define some useful functions, starting with `zeros`.
z = CUDA.functional() ? CUDA.zeros : (s...) -> zeros(Float32, s...)


ArrayType = CUDA.functional() ? CuArray : Array

# This function creates a random Gaussian force field.
function gaussian(x; σ = 0.1f0)
    n = length(x)
    xf, yf = rand(), rand()
    f = [
        exp(-(x - xf + a)^2 / σ^2 - (y - yf + a)^2 / σ^2) for x ∈ x, y in x,
        a in (-1, 0, 1), b in (-1, 0, 1)
    ] ## periodic padding
    f = reshape(sum(f; dims = (3, 4)), n, n)
    f = exp(im * rand() * 2.0f0π) * f ## Rotate f
    cat(real(f), imag(f); dims = 3)
end

# Function to chop off frequencies and multiply with scaling factor
function spectral_cutoff(u, K)
    scaling_factor = (2K)^2 / (size(u, 1) * size(u, 2))
    result = [
        u[1:K, 1:K, :] u[1:K, end-K+1:end, :]
        u[end-K+1:end, 1:K, :] u[end-K+1:end, end-K+1:end, :]
    ]
    return scaling_factor * result
end
