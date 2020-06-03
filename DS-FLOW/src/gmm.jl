# TODO: move GMM to MLToolkit.DistributionsX

module GMM

using Flux: Flux, @functor, gpu, cpu, relu
using ProgressMeter: @showprogress
using MLDataUtils: eachbatch
using Parameters: @unpack
using MLToolkit.DistributionsX: Normal
using LinearAlgebra: normalize 
using Distributions: Categorical
using Random: AbstractRNG, GLOBAL_RNG, MersenneTwister
using MLToolkit.DistributionsX: randnsimilar
using StatsFuns: logsumexp
using Statistics: mean
using Clustering: kmeans
using WeightsAndBiasLogger: wandb; const Image = wandb.Image

import Distributions: rand, logpdf

struct DiagGMM{Tm<:AbstractMatrix,Tv<:AbstractVector}
    m::Tm
    s::Tm
    mixing::Tv
    step::Ref{Int}
end

function DiagGMM(dim, n_mixtures; seed=1, rng=MersenneTwister(seed), X=nothing)
    if !isnothing(X)
        R = kmeans(X, n_mixtures; maxiter=50, display=:final)
        m = R.centers
    else
        m = rand(rng, Float32, dim, n_mixtures)
    end
    s = ones(Float32, dim, n_mixtures)
    mixing = fill(1f0 / n_mixtures, n_mixtures)
    return DiagGMM(m, s, mixing, Ref(0))
end

@functor DiagGMM

function rand(rng::AbstractRNG, gmm::DiagGMM, n::Int=1)
    z = rand(Categorical(normalize(Vector{Float64}(cpu(gmm.mixing)), 1)), n)
    return rand(Normal(gmm.m[:,z], gmm.s[:,z]))
end
rand(gmm::DiagGMM, n::Int=1) = rand(GLOBAL_RNG, gmm, n)

pxz(gmm::DiagGMM, z) = Normal(gmm.m[:,z], gmm.s[:,z])

function pxz_by_logposterior(gmm::DiagGMM, logposterior)
    posterior = cpu(exp.(logposterior))
    z = rand.(Categorical.(map(i -> normalize(Vector{Float64}(posterior[:,i]), 1), 1:size(posterior, 2))))
    return pxz(gmm, z)
end

logpdf_by_logjoint(logjoint) = dropdims(logsumexp(logjoint; dims=1); dims=1)

logpdf(gmm::DiagGMM, X) = logpdf_by_logjoint(get_logjoint(gmm, X))

function get_logjoint(gmm::DiagGMM, X)
    @unpack m, s, mixing = gmm
    dim, n = size(X)
    loglikelihood = sum(logpdf(Normal(m, s), reshape(X, dim, 1, n)), Val(:drop); dims=1)
    logprior = log.(mixing)
    logjoint = logprior .+ loglikelihood
    return logjoint
end

function get_logposterior_by_logjoint(logjoint)
    logposterior = logjoint .- logsumexp(logjoint; dims=1)
    return logposterior
end

function get_logposterior(gmm::DiagGMM, X)
    logjoint = get_logjoint(gmm, X)
    return get_logposterior_by_logjoint(logjoint)
end

function em!(gmm::DiagGMM, dataset, n_epochs, batch_size; seed=1, rng=MersenneTwister(seed))
    D_loader = eachbatch(dataset.X; size=batch_size)
    iter_last = length(D_loader)

    @unpack m, s, mixing = gmm
    dim, n_mixtures = size(m)
    # Pre-allocate memory
    m′_numerator = similar(m)
    v′_numerator = similar(s)
    total_responsibility = similar(gmm.mixing)
    @showprogress "Fitting GMM using EM" for i in 1:n_epochs
        m′_numerator .= 0
        v′_numerator .= 0
        total_responsibility .= 0
        logdensity = 0
        for (iter, x) in enumerate(D_loader)
            x = gpu(x)
            # Compute posterior
            logjoint = get_logjoint(gmm, x)
            posterior = exp.(get_logposterior_by_logjoint(logjoint))
            # Collect statistics
            m′_numerator += x * posterior'
            v′_numerator += sum((reshape(x, dim, 1, batch_size) .- m).^2 .* reshape(posterior, 1, n_mixtures, :), Val(:drop); dims=3)
            total_responsibility += sum(posterior, Val(:drop); dims=2)
            logdensity += sum(logpdf_by_logjoint(logjoint))
        end
        # Update parameters
        m .= m′_numerator ./ total_responsibility'
        s .= sqrt.(v′_numerator ./ total_responsibility')
        std_min = 1f-2 # minimal std
        s .= relu.(s .- std_min) .+ std_min
        mixing .= total_responsibility / dataset.n_data
        gmm.step[] += 1
        # Mixture means
        fig_mean = dataset.vis(cpu(gmm.m))
        # Samples from the prior
        fig_gen = dataset.vis(cpu(rand(rng, gmm, dataset.n_display)))
        # Reconstructions
        x_data = dataset.X[:,1:div(dataset.n_display, 2)]
        logposterior = get_logposterior(gmm, gpu(x_data))
        x_recon = cpu(mean(pxz_by_logposterior(gmm, logposterior)))
        fig_recon = dataset.vis((data=x_data, recon=x_recon))
        figs = (fig_mean=Image(fig_mean), fig_gen=Image(fig_gen), fig_recon=Image(fig_recon))
        @info "gmm" step=gmm.step[] logdensity=(logdensity / dataset.n_data) figs...
    end
end

export GMM, DiagGMM, em!

end # module
