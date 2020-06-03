using Bijectors: Bijectors, partition, combine, logabsdetjac, spzeros, Inversed, PartitionMask

### Tuple

 (s::Shift)(t::Tuple)             = tuple(s(first(t)),  Base.tail(t)...)
 (s::Scale)(t::Tuple)             = tuple(s(first(t)),  Base.tail(t)...)
(is::Inversed{<:Shift})(t::Tuple) = tuple(is(first(t)), Base.tail(t)...)
(is::Inversed{<:Scale})(t::Tuple) = tuple(is(first(t)), Base.tail(t)...)

### Shift

using Flux: gpu

function Bijectors._logabsdetjac_shift(a::T, x::T, ::Val{2}) where {R<:Real, T<:AbstractArray{R}}
    return zeros(R, size(a, 2)) |> gpu
end

### Scale

(b::Scale{T, 2})(x::T) where {T<:AbstractArray{<:Real,2}} = b.a .* x

function (ib::Inversed{<:Scale{T, 2}})(y::T) where {T<:AbstractArray{<:Real,2}}
    return y ./ ib.orig.a
end

function Bijectors._logabsdetjac_scale(a::T, x::T, ::Val{2}) where {R<:Real, T<:AbstractArray{R}}
    return dropdims(sum(log.(abs.(a)); dims=1); dims=1)
end

### Coupling

function (cl::CouplingLayer{B})(x::AbstractMatrix) where {B}
    x_1, x_2, x_3 = partition(cl.mask, x)
    b = B(cl.θ(x_2))
    return combine(cl.mask, b(x_1), x_2, x_3)
end

function (icl::Inversed{<:CouplingLayer{B}})(y::AbstractMatrix) where {B}
    cl = icl.orig
    y_1, y_2, y_3 = partition(cl.mask, y)
    b = B(cl.θ(y_2))
    ib = inv(b)
    return combine(cl.mask, ib(y_1), y_2, y_3)
end

function Bijectors.logabsdetjac(cl::CouplingLayer{B}, x::AbstractMatrix) where {B}
    x_1, x_2, x_3 = partition(cl.mask, x)
    b = B(cl.θ(x_2))
    return logabsdetjac(b, x_1)
end

### Tuple

function (cl::CouplingLayer{B})(t::Tuple) where {B}
    x, c = t
    x_1, x_2, x_3 = partition(cl.mask, x)
    b = B(cl.θ(vcat(x_2, c)))
    return tuple(combine(cl.mask, b(x_1), x_2, x_3), c)
end

function (icl::Inversed{<:CouplingLayer{B}})(t::Tuple) where {B}
    cl = icl.orig
    y, c = t
    y_1, y_2, y_3 = partition(cl.mask, y)
    b = B(cl.θ(vcat(y_2, c)))
    ib = inv(b)
    return tuple(combine(cl.mask, ib(y_1), y_2, y_3), c)
end

function Bijectors.logabsdetjac(cl::CouplingLayer{B}, t::Tuple) where {B}
    x, c = t
    x_1, x_2, x_3 = partition(cl.mask, x)
    b = B(cl.θ(vcat(x_2, c)))
    return logabsdetjac(b, x_1)
end

###

function Bijectors.PartitionMask(
    n::Int,
    indices_1::AbstractVector{Int},
    indices_2::AbstractVector{Int},
    indices_3::AbstractVector{Int}
)
    A_1 = spzeros(Bool, n, length(indices_1));
    A_2 = spzeros(Bool, n, length(indices_2));
    A_3 = spzeros(Bool, n, length(indices_3));

    for (i, idx) in enumerate(indices_1)
        A_1[idx, i] = true
    end

    for (i, idx) in enumerate(indices_2)
        A_2[idx, i] = true
    end

    for (i, idx) in enumerate(indices_3)
        A_3[idx, i] = true
    end

    return PartitionMask(A_1, A_2, A_3)
end

### Flux support

import Flux
Flux.@functor PartitionMask
