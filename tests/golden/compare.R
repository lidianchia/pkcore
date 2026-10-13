#!/usr/bin/env Rscript
# Compares regenerated golden outputs with the committed ones.
#
# Usage: Rscript tests/golden/compare.R committed_dir regenerated_dir [scenario_dir]
#
# Fails if the set of scenarios or engines differs, if an engine fails in one run but not
# the other, or if any concentration differs by more than RTOL relative plus ATOL times
# the largest value of that engine's output for the scenario. Outputs that a scenario
# declares as known engine issues (inaccurate closed forms, whose values are rounding
# noise and may differ between platforms) are only checked for presence and length;
# tests/test_golden.py checks that they are still wrong.

suppressPackageStartupMessages(library(jsonlite))

RTOL <- 1e-9
ATOL <- 1e-12

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(2, 3))) {
  stop("usage: compare.R committed_dir regenerated_dir [scenario_dir]")
}
committed_dir <- args[[1]]
regenerated_dir <- args[[2]]
scenario_dir <- if (length(args) == 3) args[[3]] else {
  file.path(dirname(normalizePath(committed_dir)), "scenarios")
}

known_issues <- function(file) {
  scenario <- fromJSON(file.path(scenario_dir, file), simplifyVector = FALSE)
  names(scenario$known_engine_issues)
}

names_in <- function(dir) {
  files <- list.files(dir, pattern = "\\.json$")
  sort(setdiff(files, "session.json"))
}
committed <- names_in(committed_dir)
regenerated <- names_in(regenerated_dir)
problems <- character(0)
if (!identical(committed, regenerated)) {
  problems <- c(problems, sprintf(
    "scenario sets differ: only committed: %s; only regenerated: %s",
    paste(setdiff(committed, regenerated), collapse = " "),
    paste(setdiff(regenerated, committed), collapse = " ")
  ))
}

worst <- 0
for (file in intersect(committed, regenerated)) {
  old <- fromJSON(file.path(committed_dir, file), simplifyVector = TRUE)
  new <- fromJSON(file.path(regenerated_dir, file), simplifyVector = TRUE)
  if (!identical(old$times, new$times)) problems <- c(problems, sprintf("%s: times differ", file))
  engines <- union(names(old$engines), names(new$engines))
  unstable <- known_issues(file)
  for (engine in engines) {
    a <- old$engines[[engine]]
    b <- new$engines[[engine]]
    where <- sprintf("%s / %s", file, engine)
    if (is.null(a) || is.null(b)) {
      problems <- c(problems, sprintf("%s: present in only one run", where))
    } else if (engine %in% unstable) {
      if (!is.null(a$error) || !is.null(b$error) || length(a$cp) != length(b$cp)) {
        problems <- c(problems, sprintf("%s: known-issue output changed shape", where))
      }
    } else if (!is.null(a$error) || !is.null(b$error)) {
      if (!identical(a$error, b$error)) {
        problems <- c(problems, sprintf("%s: errors differ ('%s' vs '%s')", where,
                                        format(a$error), format(b$error)))
      }
    } else {
      x <- as.numeric(a$cp)
      y <- as.numeric(b$cp)
      if (length(x) != length(y)) {
        problems <- c(problems, sprintf("%s: lengths differ", where))
        next
      }
      bound <- RTOL * abs(x) + ATOL * max(abs(x), 0)
      excess <- abs(y - x) - bound
      worst <- max(worst, max(abs(y - x) / pmax(abs(x), .Machine$double.xmin)))
      if (any(excess > 0)) {
        i <- which.max(excess)
        problems <- c(problems, sprintf("%s: value %d differs (%.17g vs %.17g)", where, i,
                                        x[[i]], y[[i]]))
      }
    }
  }
}

cat(sprintf("compared %d scenarios; largest relative difference %.3g\n",
            length(intersect(committed, regenerated)), worst))
if (length(problems) > 0) {
  cat(problems, sep = "\n")
  quit(status = 1)
}
