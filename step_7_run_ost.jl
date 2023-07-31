#=
julia
include("C:/Users/timhu/Documents/TUDelft/Courses/RP/streed_sa_pipeline/step_7_run_ost.jl")

=#

println("\033[30;1mStarted\033[0m")

total_start_time = time()

DIRECTORY = dirname(Base.source_path())

DATASET_TYPE = "binary"

using CSV
using DataFrames
using JSON
println("\033[30;1mImported modules\033[0m")

function serialize_tree(tree, node_idx, feature_meanings)
    if IAI.is_leaf(tree, node_idx)
        return "[None]"
    else
        feature_name = string(IAI.get_split_feature(tree, node_idx))

        feature_description = ""
        swap = false
        if haskey(feature_meanings, feature_name)
            feature_description = feature_meanings[feature_name]
            swap = true
        else
            attribute = "x['" * feature_name * "']"
            feature_description = "lambda x: " * attribute

            if IAI.is_parallel_split(tree, node_idx)
                feature_description *= " < " * string(IAI.get_split_threshold(tree, node_idx))
            else
                categories = IAI.get_split_categories(tree, node_idx)
                included_categores = []
                for key in keys(categories)
                    if categories[key]
                        push!(included_categores, key)
                    end
                end
                feature_description *= " in [\'" * join(included_categores, "\',\'") * "\']"
            end
        end

        lower_child = serialize_tree(tree, IAI.get_upper_child(tree, node_idx), feature_meanings)
        upper_child = serialize_tree(tree, IAI.get_lower_child(tree, node_idx), feature_meanings)

        if swap
            lower_child, upper_child = upper_child, lower_child
        end

        return "[" * feature_description *
            "," * lower_child *
            "," * upper_child * "]"
    end
end

results = []
open(DIRECTORY * "/output/settings.txt") do f
    file = "ovarian"
    max_depth = 0
    max_num_nodes = 0
    cost_complexity = 0
    hypertuning = false

    continuing = true
    warming_up = true
    settings_line = nothing

    while true
        continuing = !eof(f)

        df = CSV.read(DIRECTORY * "/datasets/" * DATASET_TYPE * "/" * file * ".txt", DataFrame, pool=true)

        events = df[!, "event"] .== 1
        times = df[!, "time"]
        X = select!(df, Not([:event, :time]))

        GC.gc()
        GC.enable(false)

        if warming_up
            println("\033[35;1mWarming up\033[0m")
        else
            println("\033[35;1mRunning " * file * ".txt {depth = " * string(max_depth) * "}\033[0m")
        end

        random_seed = 1
        object_to_fit = nothing
        start_time = end_time = 0
        already_halted_once = false
        while true
            try
                if hypertuning
                    object_to_fit = IAI.GridSearch(
                        IAI.OptimalTreeSurvivalLearner(
                            random_seed=random_seed,
                            skip_curve_fitting=false,
                            death_minbucket=0,
                        ),
                        max_depth=0:max_depth,
                    )
                else
                    object_to_fit = IAI.OptimalTreeSurvivalLearner(
                        random_seed=random_seed,
                        skip_curve_fitting=false,
                        cp=cost_complexity,
                        death_minbucket=1,
                        minbucket=1,
                        max_depth=max_depth,
                    )
                end

                start_time = time()
                IAI.fit!(object_to_fit, X, events, times)
                end_time = time()

                break
            catch ex
                if isa(ex, InterruptException)
                    if already_halted_once
                        println("\033[33;1mHalted program!\033[0m")
                        break
                    end

                    println("\033[33mKeyboard interrupt! Do it again to halt the program.\033[0m")
                    already_halted_once = true
                    continue
                end

                println("\033[31;1mAn error occurred, retrying!\033[0m")
                random_seed += 1
            end

            already_halted_once = false
        end

        GC.enable(true)

        learner = object_to_fit
        if hypertuning
            learner = IAI.get_learner(object_to_fit)
        end

        thetas = IAI.predict_hazard(learner, X)

        println(learner)

        if !warming_up
            core_name = file
            if contains(core_name, "_partition_")
                slash_idx = findfirst("/", core_name).start
                partition_idx = findfirst("_partition_", core_name).start
                core_name = core_name[(slash_idx + 1):(partition_idx - 1)]
            end
            feature_meanings = Dict()
            open(DIRECTORY * "/datasets/feature_meanings/" * core_name * ".txt") do fmf
                while !eof(fmf)
                    text = readline(fmf)
                    equals_idx = findfirst(" = ", text).start
                    key = text[1:(equals_idx - 1)]
                    value = text[(equals_idx + 3):end]
                    feature_meanings[key] = value
                end
            end

            tree_string = serialize_tree(learner, 1, feature_meanings)
            time_duration = round(end_time - start_time, digits=3)
            push!(results, (settings_line, time_duration, tree_string))

            println("\033[34;1mTime: ", time_duration, " seconds\033[0m\n")
        else
            warming_up = false
        end

        if !continuing
            break
        end

        settings_line = readline(f)
        settings = JSON.parse(settings_line)

        file = get!(settings, "file", "404")
        max_depth = get!(settings, "max-depth", 3)
        max_num_nodes = get!(settings, "max-num-nodes", 7)
        cost_complexity = get!(settings, "cost-complexity", 0)
        hypertuning = get!(settings, "mode", "direct") == "hyper"
    end
end

id = 0
open(DIRECTORY * "/output/ost_trees.csv", "w") do f
    write(f, "id;settings;time;tree\n")

    for data in results
        global id

        settings_line, time_duration, tree_string = data
        line = string(id) * ";" * settings_line * ";" * string(time_duration) * ";" * tree_string * "\n"
        write(f, line)

        id += 1
    end
end

total_end_time = time()
println("\n\033[34;1mTotal time: ", total_end_time - total_start_time, " seconds")
println("\033[32;1mDone!\033[0m")
