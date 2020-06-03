module Flow

include("normalise.jl")
using .BN

###

using Flux: Flux, Dense, relu, softplus, Chain, params, gradient, ADAM, Optimise, gpu, cpu
using Bijectors: PartitionMask, CouplingLayer, Shift, Scale, composel, forward
using MLToolkit.DistributionsX: Normal, logpdf
using Statistics: mean
using Random: MersenneTwister, shuffle
using WeightsAndBiasLogger: wandb; const Image = wandb.Image
using ProgressMeter: @showprogress
using MLDataUtils: eachbatch

import Zygote
Zygote.@nograd gpu

include("bijectors_hack.jl")

function make_cl(
    dim::Int,
    n_layers::Int, 
    nn_hidden_dim::Int, 
    is_random_mask::Bool,
    cdim::Int
)
    # Masks
    halfdim = div(dim, 2)
    masks = []
    for i in 1:n_layers
        if is_random_mask
            idcs = shuffle(1:dim)
            face, tail = idcs[1:halfdim], idcs[halfdim+1:dim]
        else
            face = filter(i ->  isodd(i), collect(1:dim))
            tail = filter(i -> iseven(i), collect(1:dim))
            if i % 2 == 1
                face, tail = tail, face
            end
        end
        mask = PartitionMask(dim, face, tail)
        push!(masks, gpu(mask))
    end

    # Modules
    scale_nns, shift_nns, bns = [], [], []
    for i in 1:n_layers
        A_1dim = size(masks[i].A_1, 2)
        A_2dim = size(masks[i].A_2, 2)
        push!(
            scale_nns,
            Chain(
                Dense(A_2dim + cdim, nn_hidden_dim, tanh),
                Dense(nn_hidden_dim, nn_hidden_dim, tanh), 
                Dense(nn_hidden_dim, A_1dim, softplus)
            ) |> gpu
        )
        push!(
            shift_nns,
            Chain(
                Dense(A_2dim + cdim, nn_hidden_dim, relu),
                Dense(nn_hidden_dim, nn_hidden_dim, relu), 
                Dense(nn_hidden_dim, A_1dim)
            ) |> gpu
        )
        push!(
            bns,
            InvertibleBatchNorm(dim) |> gpu
        )
    end
    modules = union(scale_nns, shift_nns, bns)
    
    bijector = composel(
        (
            bns[i] ∘ 
            CouplingLayer(
                θ -> Shift(θ[2]) ∘ Scale(θ[1]),
                masks[i],
                v -> tuple(scale_nns[i](v), shift_nns[i](v))
            ) for i in 1:n_layers
        )...
    )
    return bijector, modules
end

struct CouplingFlow
    base
    bijector
    modules
    opt
    step::Ref{Int}
end

function Flux.params(cf::CouplingFlow)
    ps = params()
    for m in cf.modules
        ps = params(ps..., params(m)...)
    end
    return ps
end

function CouplingFlow(dim, n_coupling_layers, nn_hidden_dim, is_random_mask, cdim, learning_rate)
    base = Normal(gpu.(tuple(zeros(Float32, dim), ones(Float32, dim)))...)
    bijector, modules = make_cl(
        dim,
        n_coupling_layers, 
        nn_hidden_dim, 
        is_random_mask,
        cdim,
    )
    opt = ADAM(learning_rate)
    return CouplingFlow(base, bijector, modules, opt, Ref(1))
end

import Distributions: logpdf

getfirst(x) = x
getfirst(t::Tuple) = first(t)

function logpdf(cf::CouplingFlow, data)
    res = forward(cf.bijector, data)
    z = getfirst(res.rv)
    lp = sum(logpdf(cf.base, z), Val(:drop); dims=1)
    logdensity = mean(lp + res.logabsdetjac)
    return logdensity
end

function evaluate(cf::CouplingFlow, data, n_display, vis; seed=1)
    rng = MersenneTwister(1)
    n_display_half = div(n_display, 2)
    function sample()
        z = rand(rng, cf.base, n_display_half)
        if data isa Tuple
            c = last(data)[:,1:n_display_half]
            data_gen = (z, c)
        else
            data_gen = z
        end
        return getfirst(inv(cf.bijector)(data_gen))
    end
    x_data = getfirst(data)[:,1:n_display_half]
    x_gen = sample()
    fig_gen = vis((data=cpu(x_data), gen=cpu(x_gen)))
    return (fig_gen=fig_gen,)
end

function ml!(cf, dataset, n_epochs, batch_size, conditional, prepare=(x -> x))
    if conditional == "label"
        D = (dataset.X, dataset.Y)
    else
        D = dataset.X
    end
    ps = params(cf)

    @showprogress for epoch in 1:n_epochs
        is_training_error = false
        for (iter, data) in enumerate(eachbatch(D; size=batch_size))
            data = prepare(data, conditional)
            local logdensity
            gs = gradient(ps) do
                logdensity = logpdf(cf, data)
                -logdensity
            end
            if isnan(logdensity) || isinf(logdensity)
                is_training_error = true
                break
            end
            Optimise.update!(cf.opt, ps, gs)
            cf.step[] += 1
            info = (logdensity=logdensity,)

            if iter % 10 == 1
                info_vis = evaluate(cf, data, dataset.n_display, dataset.vis)
                info = (info..., fig_gen=Image(info_vis.fig_gen))
            end

            @info "cflow" step=cf.step[] info...
        end
        is_training_error && break
    end
end

export CouplingFlow, ml!

end # module