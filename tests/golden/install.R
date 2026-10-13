#!/usr/bin/env Rscript
# Installs the pinned engine versions used for the golden outputs.
#
# Usage: Rscript tests/golden/install.R
#
# Packages come from a dated Posit Package Manager snapshot, by default the Linux binaries
# for Ubuntu 24.04, the base of rocker/r-ver:4.6.1. Set PKCORE_CRAN_SNAPSHOT to use another
# repository, for example https://p3m.dev/cran/2026-10-09 (source packages).

snapshot <- Sys.getenv("PKCORE_CRAN_SNAPSHOT", "https://p3m.dev/cran/__linux__/noble/2026-10-09")
pins <- c(rxode2 = "5.1.7.1", mrgsolve = "2.0.1", jsonlite = "2.0.0")

options(repos = c(CRAN = snapshot))
install.packages(names(pins), Ncpus = max(1L, parallel::detectCores()))
for (package in names(pins)) {
  found <- as.character(packageVersion(package))
  if (found != pins[[package]]) {
    stop(sprintf("%s %s is installed but %s is pinned", package, found, pins[[package]]))
  }
}
