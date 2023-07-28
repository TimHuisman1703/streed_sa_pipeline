library("hash")
library("jsonlite")
library("partykit")
library("readr")
library("survival")

directory <- getwd()
dataset_directory <- paste(directory, "/datasets", sep = "")

settings_file <- paste(directory, "/output/settings.txt", sep = "")

serialize_tree_with_features <- function(lines, root_idx, feature_meanings) {
  line <- lines[root_idx]
  depth <- lengths(regmatches(line, gregexpr("\\|   ", line)))

  if (root_idx == length(lines))
    return("[None]")

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

  if (left_idx == 0)
    return("[None]")

  feature_line <- lines[right_idx]

  start_idx <- regexpr("\\]", feature_line)[1] + 2
  end_idx <- nchar(feature_line)
  if (substring(feature_line, end_idx) == ")")
    end_idx <- regexpr("\\:[^\\:]*$", feature_line)[1] - 1
  feature_raw <- substring(feature_line, start_idx, end_idx)

  split_idx <- regexpr(" [^ ]* [^ ]*$", feature_raw)[1]
  feature_name <- substring(feature_raw, 1, split_idx - 1)
  comparison <- substring(feature_raw, split_idx)

  if (has.key(feature_name, feature_meanings)) {
    feature <- feature_meanings[[feature_name]]
  } else {
    feature <- paste("lambda x: x['", feature_name, "']", comparison, sep = "")
  }

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
  train_filename <- paste("numeric", settings["file"], sep = "/")

  # Read train data
  file_path <- paste(dataset_directory, "/", train_filename, ".txt", sep = "")
  sdata <- read.csv(file_path, stringsAsFactors = FALSE)

  # Fit and capture tree
  start_time <- Sys.time()
  tree <- ctree(
    Surv(time, event) ~ .,
    data = sdata
    # control = ctree_control(mincriterion = 0.005, minsplit = 0, minbucket = 1)
  )
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
  feature_meanings_file_path <- paste(dataset_directory, "/numeric/feature_meanings/", core_name, ".txt", sep = "")
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

  result_line <- paste("\n",
    paste(id, settings_line, end_time - start_time, tree_string, sep = ";"),
    sep = ""
  )
  result <- paste(result, result_line)

  print(tree)
  print(train_filename)

  id <- id + 1
}

sink(paste(directory, "/output/ctree_trees.csv", sep = ""))
cat(result)
sink()

print("Done!")