using DrWatson
@quickactivate "DS-GeM"

general_args = Dict(
    :dataset        => ["gaussian", "2dring", "3dring", "mnist"],
    :conditional    => ["none", "gmm", "gmm_mean", "label"],
    :n_mixtures     => [20, 40, 60, 80, 100],
)

dicts = dict_list(general_args)
paths = tmpsave(dicts)
for (p, d) in zip(paths, dicts)
    # Flitering out un-supported combinations
    # - conditional=label is only supoorted for mnist
    if d[:conditional] == "label" && d[:dataset] != "mnist"
        continue
    end
    submit = `julia $(scriptsdir("ds-gem.jl")) $p`
    run(submit)
end
