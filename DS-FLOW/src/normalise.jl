module BN

using Bijectors: Bijector, Inversed
import Bijectors: forward, logabsdetjac
using Statistics: mean

# Code adapted from Flux.jl
# Ref: https://github.com/FluxML/Flux.jl/blob/master/src/layers/normalise.jl#L93-L177
# License: https://github.com/FluxML/Flux.jl/blob/master/LICENSE.md

istraining() = false

mutable struct InvertibleBatchNorm{T1,T2,T3} <: Bijector{1}
    b       ::  T1  # bias
    logs    ::  T1  # log-scale
    m       ::  T2  # moving mean
    v       ::  T2  # moving variance
    eps     ::  T3
    mtm     ::  T3  # momentum
end

function Base.show(io::IO, l::InvertibleBatchNorm)
    print(io, "InvertibleBatchNorm($(join(size(l.b), ", ")))")
end

function InvertibleBatchNorm(
    chs::Int;
    eps::T=1f-5,
    mtm::T=1f-1,
) where {T<:AbstractFloat}
    return InvertibleBatchNorm(
        zeros(T, chs),
        zeros(T, chs),  # logs = 0 means s = 1
        zeros(T, chs),
         ones(T, chs),
        eps,
        mtm,
    )
end

using Flux: gpu

function forward(bn::InvertibleBatchNorm, x::AbstractArray)
    dims = ndims(x)
    size(x, dims - 1) == length(bn.b) ||
        error("InvertibleBatchNorm expected $(length(bn.b)) channels, got $(size(x, dims - 1))")
    channels = size(x, dims - 1)
    as = ntuple(i -> i == ndims(x) - 1 ? size(x, i) : 1, dims)
    logs = reshape(bn.logs, as...)
    s = exp.(logs)
    b = reshape(bn.b, as...)
    if istraining()
        n = div(prod(size(x)), channels)
        axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
        m = mean(x, dims = axes)
        v = sum((x .- m) .^ 2, dims = axes) ./ n
        # Update moving mean and variance
        mtm = bn.mtm
        T = eltype(bn.m)
        bn.m = (1 - mtm) .* bn.m .+ mtm .* T.(reshape(m, :))
        bn.v = (1 - mtm) .* bn.v .+ (mtm * n / (n - 1)) .* T.(reshape(v, :))
    else
        m = reshape(bn.m, as...)
        v = reshape(bn.v, as...)
    end

    rv = s .* (x .- m) ./ sqrt.(v .+ bn.eps) .+ b
    os = gpu(ones(eltype(x), size(x, dims)))
    logabsdetjac = sum(logs - log.(v .+ bn.eps) / 2) * os
    return (rv=rv, logabsdetjac=logabsdetjac)
end

logabsdetjac(bn::InvertibleBatchNorm, x) = forward(bn, x).logabsdetjac

(bn::InvertibleBatchNorm)(x) = forward(bn, x).rv

function forward(invbn::Inversed{<:InvertibleBatchNorm}, y::AbstractArray)
    @assert !istraining() "`forward(::Inversed{InvertibleBatchNorm})` is only available in test mode."
    dims = ndims(y)
    as = ntuple(i -> i == ndims(y) - 1 ? size(y, i) : 1, dims)
    bn = inv(invbn)
    s = reshape(exp.(bn.logs), as...)
    b = reshape(bn.b, as...)
    m = reshape(bn.m, as...)
    v = reshape(bn.v, as...)

    x = (y .- b) ./ s .* sqrt.(v .+ bn.eps) .+ m
    return (rv=x, logabsdetjac=-logabsdetjac(bn, x))
end

logabsdetjac(invbn::Inversed{<:InvertibleBatchNorm}, x) = forward(invbn, x).logabsdetjac

(bn::Inversed{<:InvertibleBatchNorm})(y) = forward(bn, y).rv

### Tuple

function forward(bn::Union{InvertibleBatchNorm, Inversed{<:InvertibleBatchNorm}}, t::Tuple)
    res = forward(bn, first(t))
    return (rv=tuple(res.rv, Base.tail(t)...), logabsdetjac=res.logabsdetjac)
end

export InvertibleBatchNorm

### Flux support

import Flux
Flux.trainable(bn::InvertibleBatchNorm) = (bn.b, bn.logs, bn.m, bn.v)
Flux.@functor InvertibleBatchNorm

### Zygote support

import Zygote
Zygote.@adjoint istraining() = true, _ -> nothing

end # module
