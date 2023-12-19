# This is a struct that contains the data of LES and DNS
struct Data
    t::Array{Float32,1}
    # u is usually the DNS solution
    u::Array{Array{ComplexF32, 3},1}
    # v is the LES solution (filtered DNS ubar)
    v::Array{ComplexF32, 4}
    # commutator error that can be used for derivative fitting
    c::Array{ComplexF32, 4}
    params_les::Params
    params_dns::Params
end

# Function to get the name of the file where to store data
function get_data_name(nu::Float32, les_size::Int, dns_size::Int, myseed::Int)
    return "DNS_$(dns_size)_LES_$(les_size)_nu_$(nu)_$(myseed)"
end


# This is a struct that contains the data of a trained NeuralODE
struct TrainedNODE
    θ::Any
    _closure::Any
    lhist::Array{Float32,1}
    model_name::String
    loss_name::String
end
function _get_closure(model::TrainedNODE)
    parts = split(model.model_name, "__")
    name = parts[1]
    if name == "FNO"
        ch_fno = parse.(Int, split(parts[2], "-"))
        kmax_fno = parse.(Int, split(parts[3], "-"))
        σ_fno = [eval(Symbol(name)) for name in split(parts[4], "-")]
        return create_fno_model(kmax_fno, ch_fno, σ_fno; single_timestep=false)
    elseif name == "CNN"
        r_cnn = parse.(Int, split(parts[2], "-"))
        ch_cnn = parse.(Int, split(parts[3], "-"))
        σ_cnn = [eval(Symbol(name)) for name in split(parts[4], "-")]
        b_cnn = parse.(Bool, split(parts[5], "-"))
        return create_cnn_model(r_cnn, ch_cnn, σ_cnn, b_cnn; single_timestep=false)
    else
        throw(DomainError("The model name is not recognized."))
    end
end
