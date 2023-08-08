library("hash")
library("jsonlite")
library("partykit")
library("readr")
library("survival")
library("mlr3")
library("mlr3proba")
library("mlr3extralearners")
library("mlr3tuning")
 

directory <- getwd()
dataset_type = "numeric"
TIME_OUT_IN_SECONDS <- 600

settings_file <- paste(directory, "/output/settings.txt", sep = "")

# Serialize the tree learner to a lambda-structure
#
# tree              The learner object to serialize
# node_idx          The index of the node to convert at the moment (root = 1)
# feature_meanings  The dictionary to read feature meanings from
serialize_tree_with_features <- function(lines, root_idx, feature_meanings) {
  line <- lines[root_idx]
  depth <- lengths(regmatches(line, gregexpr("\\|   ", line)))

  # Leaf node
  if (root_idx == length(lines))
    return("[None]")

  # Find left and right children
  left_idx <- 0
  right_idx <- 0
  for (i in (root_idx + 1):length(lines)) {
    pattern_descendant <- strrep("|   ", depth + 1)
    if (!startsWith(lines[i], pattern_descendant))
      break

    pattern_child <- paste(pattern_descendant, "[", sep = "")
    if (startsWith(lines[i], pattern_child)) {
      if (left_idx == 0) {
        left_idx <- i
      } else {
        right_idx <- i
        break
      }
    }
  }

  # If `left_idx` was never updated, no child exists: leaf node
  if (left_idx == 0)
    return("[None]")

  # Read feature from the right child's condition
  feature_line <- lines[right_idx]

  # Find the splitting criterium
  start_idx <- regexpr("\\]", feature_line)[1] + 2
  end_idx <- nchar(feature_line)
  if (substring(feature_line, end_idx) == ")")
    end_idx <- regexpr("\\:[^\\:]*$", feature_line)[1] - 1
  criterium <- substring(feature_line, start_idx, end_idx)

  # Split criterium in feature name and comparison
  split_idx <- regexpr(" [^ ]* [^ ]*$", criterium)[1]
  feature_name <- substring(criterium, 1, split_idx - 1)
  comparison <- substring(criterium, split_idx)

  if (has.key(feature_name, feature_meanings)) {
    # If the feature meaning is already known, use it
    feature <- feature_meanings[[feature_name]]
  } else {
    # If not, create own lambda using the comparison
    feature <- paste("lambda x: x['", feature_name, "']", comparison, sep = "")
  }

  # Serialize children
  left_child <- serialize_tree_with_features(lines, left_idx, feature_meanings)
  right_child <- serialize_tree_with_features(lines, right_idx, feature_meanings)

  return(paste("[", feature, ",", left_child, ",", right_child, "]", sep = ""))
}

id <- 0

result <- "id;settings;time;results"

settings_lines <- read_lines(settings_file)
for (settings_line in settings_lines) {
  # Read settings
  settings <- fromJSON(settings_line)
  name <- settings["file"]
  max_depth <- settings["max-depth"][[1]]

  train_filename <- paste(dataset_type, settings["file"], sep = "/")

  # Read train data
  file_path <- paste(directory, "/datasets/", train_filename, ".txt", sep = "")
  sdata <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Fit and capture tree
  start_time <- Sys.time()

  set.seed(1)

  surv.task = TaskSurv$new(id=train_filename, backend=sdata, time='time', event='event')

  surv.lrn = lrn("surv.ctree",
    maxdepth  = max_depth,
    mincriterion = to_tune(c(0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999))
  )

    instance = ti(
    task = surv.task,
    learner = surv.lrn,
    resampling = rsmp("cv", folds = 5),
    measures = msr("surv.cindex"),
    terminator = trm("run_time", secs=TIME_OUT_IN_SECONDS)
  )
  tuner = tnr("grid_search", resolution = 1)

  tuner$optimize(instance)

  surv.lrn$param_set$values = instance$result_learner_param_vals
  surv.lrn$train(surv.task)
  tree <- surv.lrn$model
  
  end_time <- Sys.time()

  tree_lines <- capture.output(print(tree))

  # Parse tree
  idx <- match(c("Fitted party:"), tree_lines)
  tree_lines <- tree_lines[(idx + 1):(length(tree_lines) - 3)]

  # Read feature meanings
  partition_idx <- regexpr("_partition_.*$", name)[1]
  core_name <- name
  if (partition_idx > -1) {
    slash_idx <- regexpr("/.*$", name)[1]
    core_name <- substring(name, slash_idx + 1, partition_idx - 1)
  }
  feature_meanings_file_path <- paste(directory, "/datasets/feature_meanings/", core_name, ".txt", sep = "")
  feature_meanings_lines <- read_lines(feature_meanings_file_path)

  # Parse feature meanings
  feature_meanings <- hash()
  for (feature_meanings_line in feature_meanings_lines) {
    split_idx <- regexpr(" = ", feature_meanings_line)[1]
    key <- substring(feature_meanings_line, 1, split_idx - 1)
    value <- substring(feature_meanings_line, split_idx + 3)
    feature_meanings[[key]] <- value
  }
  tree_string <- serialize_tree_with_features(tree_lines, 1, feature_meanings)

  # Append results
  result_line <- paste("\n",
    paste(id, settings_line, end_time - start_time, tree_string, sep = ";"),
    sep = ""
  )
  result <- paste(result, result_line)

  print(tree)
  print(train_filename)

  id <- id + 1
}

# Write trees to file
sink(paste(directory, "/output/ctree_trees.csv", sep = ""))
cat(result)
sink()

print("Done!")