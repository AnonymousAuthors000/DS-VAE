using DrWatson
@quickactivate "DS-GeM"

###

if length(ARGS) == 1
    dict = load(projectdir("_research", "tmp", ARGS[1]))
else
    dict = Dict(
        :dataset        =>  "3dring",
        :conditional    =>  "gmm_mean",
        :n_mixtures     =>  10,
    )
end

args = (
    seed           = 1,
    dataset        = dict[:dataset], #"gaussian",
    load_gmm       = false,
    continue_cflow = false,
)

@assert args.dataset in ["gaussian", "2dring", "3dring", "mnist"]

args_gmm = (
    n_epochs       = 10,
    batch_size     = 1_000,
    n_mixtures     = dict[:n_mixtures], #10,
    init_by_kmeans = true,
)

args_cflow = (
    n_epochs          = 20,
    batch_size        = 1_000,
    learning_rate     = 1f-3,
    nn_hidden_dim     = 1_000,
    n_coupling_layers = 5,
    is_random_mask    = false,
    conditional       = dict[:conditional], #"none",
)

@assert args_cflow.conditional in ["none", "gmm", "gmm_mean", "label"]
if args_cflow.conditional == "label"
    @assert args.dataset == "mnist"
end

###

using Logging, WeightsAndBiasLogger

logger = WBLogger(; project="ds-gem")
config!(logger, args)
config!(logger, :gmm => args_gmm)
config!(logger, :cflow => args_cflow)

###

using MLToolkit.Datasets: Dataset

function get_dataset(name; kwargs...)
    args = (60_000,)
    if name == "mnist"
        @assert !(:is_flatten in keys(kwargs))
        kwargs = (kwargs..., is_flatten=true)
    end
    return Dataset(name, args...; kwargs...)
end

dataset = get_dataset(args.dataset; seed=args.seed)

###

using Flux: gpu, cpu, params, loadparams!

###

include(srcdir("gmm.jl"))
using .GMM
savepath_gmm = datadir("results", args.dataset, savename(args_gmm, "bson"))

if args.load_gmm
    gmm = load(savepath_gmm)[:model]
else 
    gmm = DiagGMM(
        dataset.dim, args_gmm.n_mixtures; seed=args.seed, X=(args_gmm.init_by_kmeans ? dataset.X : nothing)
    ) |> gpu

    with_logger(logger) do
        em!(gmm, dataset, args_gmm.n_epochs, args_gmm.batch_size)
    end

    gmm = gmm |> cpu

    tagsave(savepath_gmm, Dict(:model => gmm); safe=true)
end

###

cdim_dict = Dict(
    "none"     => 0, 
    "gmm"      => dataset.dim, 
    "gmm_mean" => dataset.dim, 
    "label"    => 10,
)

include(srcdir("flow.jl"))
using .Flow
savepath_cflow = datadir("results", args.dataset, savename(args_cflow, "bson"))

cflow = CouplingFlow(
    dataset.dim, 
    args_cflow.n_coupling_layers, 
    args_cflow.nn_hidden_dim, 
    args_cflow.is_random_mask, 
    cdim_dict[args_cflow.conditional], 
    args_cflow.learning_rate,
)

using .GMM: get_logposterior, pxz_by_logposterior, mean

function get_prepare(gmm)
    function prepare(data, conditional)
        if conditional == "none"
            data = gpu(data)
        else
            if conditional != "label"
                x = data
                logposterior = get_logposterior(gmm, x)
                pxz = pxz_by_logposterior(gmm, logposterior)
                if conditional == "gmm"
                    y = rand(pxz)
                end
                if conditional == "gmm_mean"
                    y = mean(pxz)
                end
                data = (x, y)
            end
            data = gpu.(data)
        end
        return data
    end
    return prepare
end

if args.continue_cflow
    cflowfile = load(savepath_cflow)
    loadparams!(cflow, cflowfile[:weights])
    cflow.step[] = cflowfile[:step]
end

with_logger(logger) do
    ml!(cflow, dataset, args_cflow.n_epochs, args_cflow.batch_size, args_cflow.conditional, get_prepare(gmm))
end

tagsave(savepath_cflow, Dict(:weights => Array.(params(cflow)), :step => cflow.step[]); safe=true)
