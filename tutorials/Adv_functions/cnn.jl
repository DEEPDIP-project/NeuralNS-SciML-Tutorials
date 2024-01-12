using Flux

function find_padding_size(CNN,test_size = 100)
    dims = length(size(CNN[1].weight)) - 2
    input_channels = size(CNN[1].weight)[dims + 1]
    test_input = zeros(Tuple([[test_size for i in 1:dims]...,input_channels,1]))
    test_output = CNN(test_input)
    required_padding = ([size(test_input)...] .- [size(test_output)...])[1:dims]
    return Tuple(Int.(required_padding ./ 2))
end

function conv_NN(widths,channels,strides = 0,bias = true)
    if strides == 0
        strides = ones(Int,size(widths)[1])
    end
    pad = Tuple(zeros(Int,length(widths[1])))
    storage = []
    for i in 1:size(widths)[1]
        kernel_size = Tuple(2* [widths[i]...] .+ 1)
        if i == size(widths)[1]
            storage = [storage;Flux.Conv(kernel_size, channels[i]=>channels[i+1],stride = strides[i],pad = pad,bias = bias)]
        else

            storage = [storage;Flux.Conv(kernel_size, channels[i]=>channels[i+1],stride = strides[i],pad = pad,relu,bias = bias)]
        end
    end
    return Flux.Chain((i for i in storage)...)
end


function gen_channel_mask(data,channel)
    dims = length(size(data)) - 2
    number_of_channels = size(data)[end-1]
    channel_mask = stop_gradient() do
        zeros(size(data)[1:end-1])
    end
    stop_gradient() do
        channel_mask[[(:) for i in 1:dims]...,channel] .+= 1
    end
    return channel_mask
end

function construct_corner_mask(N,pad_size)
    dims = length(N)
    corner_mask = zeros(N)
    for i in 1:dims
        original_dims = collect(1:length(size(corner_mask)))
        new_dims = copy(original_dims)
        new_dims[1] = original_dims[i]
        new_dims[i] =   1

        corner_mask = permutedims(corner_mask,new_dims)
        pad_start = corner_mask[(end-pad_size[i]+1):end,[(:) for j in 1:(dims-1)]...]
        pad_end = corner_mask[1:pad_size[i],[(:) for j in 1:(dims-1)]...]




        if i == dims
            pad_start = permutedims(pad_start,new_dims)
            pad_end = permutedims(pad_end,new_dims)

            for j in 1:dims-1
                select_start = [(:) for k in 1:(j-1)]
                select_end = [(:) for k in j:(dims-1)]
                pad_start[select_start...,1:pad_size[j],select_end...] .+= 1
                pad_start[select_start...,end-pad_size[j]+1:end,select_end...] .+= 1
                pad_end[select_start...,1:pad_size[j],select_end...] .+= 1
                pad_end[select_start...,end-pad_size[j]+1:end,select_end...] .+= 1

            end

            pad_start = permutedims(pad_start,new_dims)
            pad_end = permutedims(pad_end,new_dims)
            corner_mask = cat([pad_start,corner_mask,pad_end]...,dims = 1)
        else

            corner_mask = cat([pad_start,corner_mask,pad_end]...,dims = 1)

        end
        corner_mask = permutedims(corner_mask,new_dims)

    end
    return corner_mask
end