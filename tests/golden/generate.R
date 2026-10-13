#!/usr/bin/env Rscript
# Cross-engine golden outputs for pkcore (verification level V3).
#
# Reads every scenario in tests/golden/scenarios/*.json, simulates it with
#   rxode2_lincmt     rxode2, closed-form linCmt() solution
#   rxode2_ode        rxode2, ODE system (liblsoda, atol = rtol = 1e-12)
#   mrgsolve_pkmodel  mrgsolve, closed-form $PKMODEL solution
#   mrgsolve_ode      mrgsolve, ODE system (atol = rtol = 1e-12)
# and writes one JSON file per scenario plus session.json and sessionInfo.txt.
#
# Usage: Rscript tests/golden/generate.R [scenario_dir] [output_dir]
#
# Conventions shared with pkcore: doses into the central compartment (IV bolus,
# zero-order infusion with rate = amount / duration) or into a depot (oral) that empties
# at the first-order rate KA; bioavailability and lag time are defined for the depot
# only, so they apply to oral doses only. Concentrations are central amount / central
# volume. Observation and dose records at the same time are passed in the order given by
# the scenario ("dose_first" or "observation_first"); mrgsolve honours that order,
# rxode2 always applies the dose first.

suppressPackageStartupMessages({
  library(jsonlite)
  library(rxode2)
  library(mrgsolve)
})

args <- commandArgs(trailingOnly = TRUE)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- dirname(normalizePath(script))
scenario_dir <- if (length(args) >= 1) args[[1]] else file.path(here, "scenarios")
output_dir <- if (length(args) >= 2) args[[2]] else file.path(here, "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

ODE_TOLERANCE <- 1e-12

# --- models ------------------------------------------------------------------------

rx_lin1 <- rxode2({
  CL <- pCL
  V <- pV
  KA <- pKA
  f(depot) <- pF
  alag(depot) <- pTLAG
  cp <- linCmt()
})

rx_lin2 <- rxode2({
  CL <- pCL
  V1 <- pV1
  Q <- pQ
  V2 <- pV2
  KA <- pKA
  f(depot) <- pF
  alag(depot) <- pTLAG
  cp <- linCmt()
})

rx_ode <- rxode2({
  d/dt(depot) <- -pKA * depot
  d/dt(central) <- pKA * depot - (pCL + pQ) / pV1 * central + pQ / pV2 * periph
  d/dt(periph) <- pQ / pV1 * central - pQ / pV2 * periph
  f(depot) <- pF
  alag(depot) <- pTLAG
  cp <- central / pV1
})

mrg_pk1 <- mcode("pkcore_golden_pk1", '
$PARAM CL = 1, V = 1, KA = 1, FF = 1, TLAG = 0
$PKMODEL cmt = "DEPOT CENT", depot = TRUE
$MAIN
F_DEPOT = FF;
ALAG_DEPOT = TLAG;
$TABLE
capture CP = CENT / V;
', quiet = TRUE)

mrg_pk2 <- mcode("pkcore_golden_pk2", '
$PARAM CL = 1, V2 = 1, Q = 1, V3 = 1, KA = 1, FF = 1, TLAG = 0
$PKMODEL cmt = "DEPOT CENT PERIPH", depot = TRUE
$MAIN
F_DEPOT = FF;
ALAG_DEPOT = TLAG;
$TABLE
capture CP = CENT / V2;
', quiet = TRUE)

mrg_ode <- mcode("pkcore_golden_ode", '
$PARAM CL = 1, V1 = 1, Q = 0, V2 = 1, KA = 1, FF = 1, TLAG = 0
$CMT DEPOT CENT PERIPH
$MAIN
F_DEPOT = FF;
ALAG_DEPOT = TLAG;
$ODE
dxdt_DEPOT = -KA * DEPOT;
dxdt_CENT = KA * DEPOT - (CL + Q) / V1 * CENT + Q / V2 * PERIPH;
dxdt_PERIPH = Q / V1 * CENT - Q / V2 * PERIPH;
$TABLE
capture CP = CENT / V1;
', quiet = TRUE)
mrg_ode <- update(mrg_ode, atol = ODE_TOLERANCE, rtol = ODE_TOLERANCE, maxsteps = 1e6)

# --- scenario -> engine inputs ------------------------------------------------------

two_compartments <- function(model) model$compartments == 2

# Dose and observation records, ordered by time; at equal times the scenario decides
# whether dose or observation records come first.
records <- function(scenario) {
  doses <- scenario$doses
  dose <- data.frame(
    time = vapply(doses, function(d) d$time, numeric(1)),
    amt = vapply(doses, function(d) d$amount, numeric(1)),
    route = vapply(doses, function(d) d$route, character(1)),
    rate = vapply(doses, function(d) {
      if (d$route == "iv_infusion") d$amount / d$duration else 0
    }, numeric(1)),
    evid = 1,
    stringsAsFactors = FALSE
  )
  times <- unlist(scenario$observations$times)
  obs <- data.frame(time = times, amt = 0, route = "observation", rate = 0, evid = 0,
                    stringsAsFactors = FALSE)
  all <- rbind(dose, obs)
  dose_first <- scenario$observations$order == "dose_first"
  rank <- ifelse(all$evid == 1, if (dose_first) 0 else 1, if (dose_first) 1 else 0)
  all[order(all$time, rank, seq_len(nrow(all))), ]
}

rxode2_data <- function(recs) {
  data.frame(
    id = 1, time = recs$time, evid = recs$evid, amt = recs$amt, rate = recs$rate,
    cmt = ifelse(recs$route == "oral", "depot", "central"), stringsAsFactors = FALSE
  )
}

mrgsolve_data <- function(recs) {
  data.frame(
    ID = 1, time = recs$time, evid = recs$evid, amt = recs$amt, rate = recs$rate,
    cmt = ifelse(recs$evid == 0, 0, ifelse(recs$route == "oral", 1, 2))
  )
}

run_rxode2 <- function(model, params, recs, times) {
  out <- rxSolve(model, params, rxode2_data(recs), atol = ODE_TOLERANCE, rtol = ODE_TOLERANCE,
                 maxsteps = 1e6, returnType = "data.frame")
  stopifnot(isTRUE(all.equal(out$time, times, tolerance = 0)))
  out$cp
}

run_mrgsolve <- function(model, recs, times) {
  out <- as.data.frame(mrgsim(model, data = mrgsolve_data(recs), obsonly = TRUE))
  stopifnot(isTRUE(all.equal(out$time, times, tolerance = 0)))
  out$CP
}

simulate <- function(scenario) {
  m <- scenario$model
  recs <- records(scenario)
  times <- unlist(scenario$observations$times)
  ka <- m$ka
  if (two_compartments(m)) {
    lin <- rx_lin2
    lin_params <- c(pCL = m$cl, pV1 = m$v1, pQ = m$q, pV2 = m$v2, pKA = ka, pF = m$f,
                    pTLAG = m$tlag)
    ode_params <- lin_params
    pk_model <- param(mrg_pk2, CL = m$cl, V2 = m$v1, Q = m$q, V3 = m$v2, KA = ka, FF = m$f,
                      TLAG = m$tlag)
    mrg_ode_model <- param(mrg_ode, CL = m$cl, V1 = m$v1, Q = m$q, V2 = m$v2, KA = ka,
                           FF = m$f, TLAG = m$tlag)
  } else {
    lin <- rx_lin1
    lin_params <- c(pCL = m$cl, pV = m$v, pKA = ka, pF = m$f, pTLAG = m$tlag)
    ode_params <- c(pCL = m$cl, pV1 = m$v, pQ = 0, pV2 = 1, pKA = ka, pF = m$f,
                    pTLAG = m$tlag)
    pk_model <- param(mrg_pk1, CL = m$cl, V = m$v, KA = ka, FF = m$f, TLAG = m$tlag)
    mrg_ode_model <- param(mrg_ode, CL = m$cl, V1 = m$v, Q = 0, V2 = 1, KA = ka, FF = m$f,
                           TLAG = m$tlag)
  }
  engines <- list(
    rxode2_lincmt = function() run_rxode2(lin, lin_params, recs, times),
    rxode2_ode = function() run_rxode2(rx_ode, ode_params, recs, times),
    mrgsolve_pkmodel = function() run_mrgsolve(pk_model, recs, times),
    mrgsolve_ode = function() run_mrgsolve(mrg_ode_model, recs, times)
  )
  results <- lapply(engines, function(engine) {
    tryCatch(list(cp = engine()), error = function(e) list(error = conditionMessage(e)))
  })
  list(scenario = scenario$name, times = times, engines = results)
}

files <- sort(list.files(scenario_dir, pattern = "\\.json$", full.names = TRUE))
if (length(files) == 0) stop("no scenarios found in ", scenario_dir)
for (file in files) {
  scenario <- fromJSON(file, simplifyVector = FALSE)
  result <- simulate(scenario)
  target <- file.path(output_dir, basename(file))
  writeLines(toJSON(result, auto_unbox = TRUE, digits = I(17), pretty = TRUE), target)
  status <- vapply(result$engines, function(r) if (is.null(r$error)) "ok" else "error",
                   character(1))
  message(sprintf("%-40s %s", scenario$name, paste(names(status), status, collapse = ", ")))
}

packages <- c("rxode2", "mrgsolve", "jsonlite", "rxode2ll", "lotri", "dparser", "PreciseSums",
              "Rcpp", "RcppArmadillo", "data.table")
session <- list(
  r_version = R.version.string,
  platform = R.version$platform,
  repos = unname(getOption("repos")[["CRAN"]]),
  ode_tolerance = ODE_TOLERANCE,
  packages = as.list(vapply(packages, function(p) as.character(packageVersion(p)),
                            character(1)))
)
writeLines(toJSON(session, auto_unbox = TRUE, pretty = TRUE), file.path(output_dir, "session.json"))
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
