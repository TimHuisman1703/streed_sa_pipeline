#=
julia
include("C:/Users/timhu/Documents/TUDelft/Courses/RP/streed_sa_pipeline/step_7_run_ost.jl")

=#

println("\033[30;1mStarted\033[0m")

total_start_time = time()

DIRECTORY = dirname(Base.source_path())

using CSV
using DataFrames
using JSON
println("\033[30;1mImported modules\033[0m")

function serialize_tree(tree, node_idx)
    if IAI.is_leaf(tree, node_idx)
        return "[None]"
    else
        attribute = "x['" * string(IAI.get_split_feature(tree, node_idx)) * "']"
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

        return "[" * feature_description *
            "," * serialize_tree(tree, IAI.get_upper_child(tree, node_idx)) *
            "," * serialize_tree(tree, IAI.get_lower_child(tree, node_idx)) * "]"
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
        continuing = !eof(f);

        df = CSV.read(DIRECTORY * "/datasets/original/" * file * ".txt", DataFrame, pool=true)

        events = df[!, "event"] .== 1
        times = df[!, "time"]
        X = select!(df, Not([:event, :time]))

        object_to_fit = nothing
        if hypertuning
            object_to_fit = IAI.GridSearch(
                IAI.OptimalTreeSurvivalLearner(
                    missingdatamode=:separate_class,
                    random_seed=1,
                    skip_curve_fitting=false,
                    death_minbucket=0,
                ),
                max_depth=0:max_depth,
            )
        else
            object_to_fit = IAI.OptimalTreeSurvivalLearner(
                missingdatamode=:separate_class,
                random_seed=1,
                skip_curve_fitting=false,
                cp=cost_complexity,
                death_minbucket=0,
                max_depth=max_depth,
            )
        end

        GC.gc()
        GC.enable(false)

        if warming_up
            println("\033[35;1mWarming up\033[0m")
        else
            println("\033[35;1mRunning " * file * ".txt {depth = " * string(max_depth) * "}\033[0m")
        end

        start_time = time()
        IAI.fit!(object_to_fit, X, events, times)
        end_time = time()

        GC.enable(true)

        learner = object_to_fit
        if hypertuning
            learner = IAI.get_learner(object_to_fit)
        end

        thetas = IAI.predict_hazard(learner, X)

        println(learner)

        if !warming_up
            tree_string = serialize_tree(learner, 1)
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
