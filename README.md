# cassava-yield-starch-analysis
Reproducibility pipeline for analyzing management factors affecting cassava yield and starch content in the western Brazilian Cerrado - Full R pipeline for statistical analysis, machine learning, and boundary modeling of cassava yield and starch.

# Reproducibility Code for Cassava Yield and Starch Analysis

This repository contains the full reproducibility pipeline for the article:

**Assessing management factors limiting yield and starch content of cassava in the western Brazilian Cerrado**  
(Zebalho et al., 2024)

The goal of this repository is to ensure transparency, traceability, and open access to all analytical steps used to reproduce the study’s main findings.

Reproducibility code for:
"Assessing management factors limiting yield and starch content of cassava in the western Brazilian Cerrado" (Zebalho et al., 2024). See uploaded paper. :contentReference[oaicite:2]{index=2}

Files:
- analysis_pipeline.R  : Main pipeline. Requires cassava_survey.csv in same folder.
- simulate_data.R     : Generates a simulated cassava_survey.csv for tests.
- cassava_survey.csv  : (user-provided) CSV with survey data.
- outputs/             : (created by scripts) figures, model objects, summaries.

Instructions:
1. Install R >= 4.0 and Rtools (if on Windows).
2. In an R session, set working directory to repository folder.
3. Optionally run `simulate_data.R` to create a test dataset:
     source("simulate_data.R")
4. Run:
     source("analysis_pipeline.R")
5. Check produced PNGs and .txt/.rds files.

Required R packages: tidyverse, rpart, rpart.plot, caret, broom, lubridate, ggpubr, nlstools, segmented, readr

Notes & assumptions:
- This pipeline assumes a CSV column naming scheme (see top of script).
- The boundary (bilinear plateau) is approximated using segmented regression. Alternate nonlinear forms can be fitted with `nls` if desired.
- The regression tree uses `rpart` and default hyperparameters; pruning and cross-validation options are available if finer control is required.
- The code computes yield gap with Yp = 63.2 Mg ha^-1 (value cited in the paper). See yield_gap.txt for results.

---

## 📁 Repository Structure

├── analysis_pipeline.R # Main analysis code (statistics, ML, boundary models)
├── simulate_data.R # Optional: generates a synthetic cassava_survey.csv for testing
├── cassava_survey.csv # User-provided dataset (not included in repo)
├── outputs/ # Automatically generated: figures, tables, model results
├── LICENSE # MIT License (open-source)
└── README.md # Documentation


# analysis_pipeline.R
# Reproducibility pipeline for:
# "Assessing management factors limiting yield and starch content of cassava..."
# (Zebalho et al., Agronomy Journal 2024). See: uploaded paper. :contentReference[oaicite:1]{index=1}
#
# Usage:
# 1) Install required packages (see README.md).
# 2) Place cassava_survey.csv in working directory (or run simulate_data.R to create one).
# 3) Run this script: source("analysis_pipeline.R")

# ---------------------------
# 0. Settings and libraries
# ---------------------------
required_packages <- c("tidyverse","rpart","rpart.plot","caret","broom",
                       "lubridate","ggpubr","nlstools","segmented","readr")
install_if_missing <- function(pkgs){
  new <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
  if(length(new)) install.packages(new, repos="https://cloud.r-project.org")
}
install_if_missing(required_packages)

library(tidyverse)
library(rpart); library(rpart.plot)
library(caret)
library(broom)
library(lubridate)
library(ggpubr)
library(nlstools)
library(segmented)
library(readr)

set.seed(123)

# ---------------------------
# 1. Load data
# ---------------------------
# Expect CSV with at least columns:
# yield_Mg_ha, starch_g_per_5kg, planting_date (YYYY-MM-DD or DOY), variety,
# P2O5_kg_ha, K2O_kg_ha, weed_control_sprays, crop_rotation (Yes/No), buds_count
# Example file name: cassava_survey.csv

data_file <- "cassava_survey.csv"
if(!file.exists(data_file)){
  stop(paste("Data file", data_file, "not found. Run simulate_data.R or provide cassava_survey.csv"))
}
df <- read_csv(data_file, show_col_types = FALSE)

# Standardize column names (lowercase)
names(df) <- tolower(names(df))

# Convert planting_date to Date and DOY
if("planting_date" %in% names(df)){
  # Try parsing different formats
  df <- df %>%
    mutate(planting_date = as.Date(planting_date)) %>%
    mutate(doy = yday(planting_date))
} else if("doy" %in% names(df)){
  df <- df %>% mutate(doy = as.numeric(doy))
} else {
  stop("No planting_date or doy column found in the dataset.")
}

# Basic required columns check
req_cols <- c("yield_mg_ha","starch_g_per_5kg","doy","variety")
missing_cols <- setdiff(req_cols, names(df))
if(length(missing_cols)>0){
  stop(paste("Missing required columns:", paste(missing_cols, collapse=", ")))
}

# ---------------------------
# 2. Create high/low groups (tertiles as described)
# ---------------------------
# Paper: 20% above average (HY) and 20% below average (LY)
ya <- mean(df$yield_mg_ha, na.rm=TRUE)
high_cut <- quantile(df$yield_mg_ha, probs=0.8, na.rm=TRUE)
low_cut  <- quantile(df$yield_mg_ha, probs=0.2, na.rm=TRUE)

df <- df %>%
  mutate(yield_group = case_when(
    yield_mg_ha >= high_cut ~ "HY",
    yield_mg_ha <= low_cut  ~ "LY",
    TRUE ~ "MID"
  ))

# For starch groups: HS if starch >= 500 g/5kg as paper used
df <- df %>% mutate(starch_group = ifelse(starch_g_per_5kg >= 500, "HS", "LS"))

# ---------------------------
# 3. Descriptive stats & group comparisons
# ---------------------------

# Summary table by yield_group
summary_by_group <- df %>%
  group_by(yield_group) %>%
  summarise(n = n(),
            mean_yield = mean(yield_mg_ha, na.rm=TRUE),
            median_doy  = median(doy, na.rm=TRUE),
            mean_P2O5   = mean(ifelse(!is.na(p2o5_kg_ha), p2o5_kg_ha, NA), na.rm=TRUE),
            mean_K2O    = mean(ifelse(!is.na(k2o_kg_ha), k2o_kg_ha, NA), na.rm=TRUE),
            mean_weed_cs = mean(ifelse(!is.na(weed_control_sprays), weed_control_sprays, NA), na.rm=TRUE)
  )

write_csv(summary_by_group, "summary_by_yield_group.csv")

# t-test or wilcoxon comparing yields (HY vs LY)
hy_vals <- df %>% filter(yield_group=="HY") %>% pull(yield_mg_ha)
ly_vals <- df %>% filter(yield_group=="LY") %>% pull(yield_mg_ha)

ttest_yield <- tryCatch(t.test(hy_vals, ly_vals), error=function(e) e)
if(inherits(ttest_yield,"error")) ttest_yield <- wilcox.test(hy_vals, ly_vals)

# Chi-square for categorical vars (example: crop_rotation)
if("crop_rotation" %in% names(df)){
  tbl_cr <- table(df$yield_group, df$crop_rotation)
  chi_crop_rotation <- chisq.test(tbl_cr)
} else chi_crop_rotation <- NULL

# Save tests
sink("stat_tests.txt")
cat("Yield summary by group:\n")
print(summary_by_group)
cat("\nYield HY vs LY test:\n")
print(ttest_yield)
cat("\nChi-square crop_rotation (if available):\n")
print(chi_crop_rotation)
sink()

# ---------------------------
# 4. Regression tree analysis (rpart), caret split 80/20
# ---------------------------
# Prepare predictors; simple formula using management variables
# Adjust to the real columns available in your CSV

predictors <- c()
possible_predictors <- c("variety","p2o5_kg_ha","k2o_kg_ha","doy","weed_control_sprays","crop_rotation","buds_count")
predictors <- intersect(possible_predictors, names(df))

# Convert categorical predictors to factors
df <- df %>% mutate(across(all_of(c("variety","crop_rotation")), as.factor))

# Remove rows with missing response
df_rpart <- df %>% filter(!is.na(yield_mg_ha))

# Split
train_index <- createDataPartition(df_rpart$yield_mg_ha, p = 0.8, list = FALSE)
train <- df_rpart[train_index, ]
test  <- df_rpart[-train_index, ]

# Fit rpart
formula_yield <- as.formula(paste("yield_mg_ha ~", paste(predictors, collapse = " + ")))
rpart_fit <- rpart(formula_yield, data=train, method="anova", na.action = na.rpart)

# Plot and save
png("rpart_yield_tree.png", width=1200, height=800)
rpart.plot(rpart_fit, main="Regression tree for cassava yield")
dev.off()

# Predict and RMSE
pred_test <- predict(rpart_fit, newdata=test)
rmse_yield <- RMSE(pred_test, test$yield_mg_ha)

# Save model summary
sink("rpart_yield_summary.txt")
print(summary(rpart_fit))
cat("\nValidation RMSE (yield):", rmse_yield, "\n")
sink()

# Repeat for starch content
df_rpart_s <- df %>% filter(!is.na(starch_g_per_5kg))
train_index_s <- createDataPartition(df_rpart_s$starch_g_per_5kg, p = 0.8, list = FALSE)
train_s <- df_rpart_s[train_index_s, ]
test_s  <- df_rpart_s[-train_index_s, ]

predictors_s <- intersect(possible_predictors, names(df_rpart_s))
formula_starch <- as.formula(paste("starch_g_per_5kg ~", paste(predictors_s, collapse = " + ")))
rpart_fit_s <- rpart(formula_starch, data=train_s, method="anova", na.action = na.rpart)

png("rpart_starch_tree.png", width=1200, height=800)
rpart.plot(rpart_fit_s, main = "Regression tree for cassava starch content")
dev.off()

pred_test_s <- predict(rpart_fit_s, newdata=test_s)
rmse_starch <- RMSE(pred_test_s, test_s$starch_g_per_5kg)

sink("rpart_starch_summary.txt")
print(summary(rpart_fit_s))
cat("\nValidation RMSE (starch):", rmse_starch, "\n")
sink()

# ---------------------------
# 5. Boundary function (bilinear plateau) for yield vs planting date
# ---------------------------
# The paper describes a bilinear plateau (two-segment) model:
# strategy: fit piecewise linear (broken-stick) with plateau (use segmented + manual plateau)
# A simple approach: fit an asymptotic bilinear using 'segmented' on lm and then extract slope.

# Fit linear model
df_bp <- df %>% filter(!is.na(yield_mg_ha), !is.na(doy))
lm_yield <- lm(yield_mg_ha ~ doy, data = df_bp)
# Use segmented to find breakpoint
seg_yield <- tryCatch(segmented(lm_yield, seg.Z = ~doy, psi = median(df_bp$doy, na.rm=TRUE)), error=function(e) NULL)

# If segmented successful, extract breakpoint and slopes
if(!is.null(seg_yield)){
  bp_est <- summary(seg_yield)$psi[, "Est."]
  slopes <- slope(seg_yield)$doy
  yield_bp_info <- list(breakpoint = bp_est, slopes = slopes)
} else {
  yield_bp_info <- list(error="segmented fit failed; consider nls or manual optimization.")
}

# Save yield boundary info
sink("boundary_yield.txt")
print(yield_bp_info)
sink()

# Plot yield vs doy with fitted line
p1 <- ggplot(df_bp, aes(x=doy, y=yield_mg_ha)) +
  geom_point(alpha=0.6) +
  labs(x="Day of year (planting date)", y="Yield (Mg ha-1)") +
  theme_minimal()

if(!is.null(seg_yield)){
  newd <- data.frame(doy = seq(min(df_bp$doy, na.rm=TRUE), max(df_bp$doy, na.rm=TRUE), length.out = 200))
  newd$pred <- predict(seg_yield, newdata = newd)
  p1 <- p1 + geom_line(data=newd, aes(x=doy, y=pred))
}

ggsave("yield_vs_doy_boundary.png", p1, width=8, height=5)

# ---------------------------
# 6. Boundary for starch vs planting date
# ---------------------------
df_bp_s <- df %>% filter(!is.na(starch_g_per_5kg), !is.na(doy))
lm_starch <- lm(starch_g_per_5kg ~ doy, data = df_bp_s)
seg_starch <- tryCatch(segmented(lm_starch, seg.Z = ~doy, psi = median(df_bp_s$doy, na.rm=TRUE)), error=function(e) NULL)

if(!is.null(seg_starch)){
  bp_est_s <- summary(seg_starch)$psi[, "Est."]
  slopes_s <- slope(seg_starch)$doy
  starch_bp_info <- list(breakpoint = bp_est_s, slopes = slopes_s)
} else {
  starch_bp_info <- list(error="segmented fit failed; consider nls or manual optimization.")
}

sink("boundary_starch.txt")
print(starch_bp_info)
sink()

p2 <- ggplot(df_bp_s, aes(x=doy, y=starch_g_per_5kg)) +
  geom_point(alpha=0.6) +
  labs(x="Day of year (planting date)", y="Starch content (g per 5 kg)") +
  theme_minimal()

if(!is.null(seg_starch)){
  newd2 <- data.frame(doy = seq(min(df_bp_s$doy, na.rm=TRUE), max(df_bp_s$doy, na.rm=TRUE), length.out = 200))
  newd2$pred <- predict(seg_starch, newdata = newd2)
  p2 <- p2 + geom_line(data=newd2, aes(x=doy, y=pred))
}

ggsave("starch_vs_doy_boundary.png", p2, width=8, height=5)

# ---------------------------
# 7. Yield gap calculation relative to Yp = 63.2 Mg/ha
# ---------------------------
Yp <- 63.2
Ya <- mean(df$yield_mg_ha, na.rm=TRUE)
yield_gap <- Yp - Ya

cat(sprintf("Yield potential (Yp) = %.2f Mg ha-1\nMean observed yield (Ya) = %.2f Mg ha-1\nYield gap = %.2f Mg ha-1\n",
            Yp, Ya, yield_gap), file = "yield_gap.txt")

# ---------------------------
# 8. Save models and objects
# ---------------------------
saveRDS(rpart_fit, "rpart_yield_model.rds")
saveRDS(rpart_fit_s, "rpart_starch_model.rds")
if(!is.null(seg_yield)) saveRDS(seg_yield, "segmented_yield_model.rds")
if(!is.null(seg_starch)) saveRDS(seg_starch, "segmented_starch_model.rds")

# ---------------------------
# 9. Final report snippet
# ---------------------------
report <- list(
  mean_yield = Ya,
  yield_gap = yield_gap,
  rpart_yield_rmse = rmse_yield,
  rpart_starch_rmse = rmse_starch
)
write_json(report, "analysis_report_summary.json", pretty=TRUE)

cat("Analysis finished. Outputs produced in working directory:\n - rpart models and plots\n - boundary fits and plots\n - summary tables and test results\nCheck *_summary.txt, *_boundary.txt, and saved PNGs.\n")


# simulate_data.R
# Generate a simulated cassava_survey.csv with realistic structure
library(tidyverse); library(lubridate); set.seed(42)

n <- 300
varieties <- c("B36","B420","BCS","I14","IU","P","B","FB","FM","I15","I90","IP","NM","O","SG")
planting_dates <- sample(seq(as.Date("2020-05-01"), as.Date("2020-09-30"), by="day"), n, replace=TRUE)
doy <- yday(planting_dates)

# Simulate yields centered ~18.6 with wide spread; include effects
yield_mg_ha <- rnorm(n, mean=18.6, sd=8)
# Add effect: earlier planting -> higher yields
yield_mg_ha <- yield_mg_ha + (200 - doy) * 0.02
# variety effect: modern varieties have +6 Mg
yield_mg_ha <- yield_mg_ha + ifelse(varieties %in% c("B36","B420","BCS","I14","IU","P")[sample(1:length(varieties), n, replace=TRUE)], 6, 0)

starch_g_per_5kg <- rnorm(n, mean=500, sd=80)
starch_g_per_5kg <- starch_g_per_5kg + ifelse(doy >= 146 & doy <= 230, 30, -10) # planting window effect

p2o5 <- rpois(n, lambda=70)
k2o  <- rpois(n, lambda=60)
weed_control_sprays <- rpois(n, lambda=4)
crop_rotation <- sample(c("Yes","No"), n, replace=TRUE, prob=c(0.5,0.5))
buds_count <- sample(4:10, n, replace=TRUE)

df_sim <- tibble(
  yield_mg_ha = pmax(0.5, yield_mg_ha),
  starch_g_per_5kg = pmax(200, starch_g_per_5kg),
  planting_date = planting_dates,
  doy = doy,
  variety = sample(varieties, n, replace=TRUE),
  p2o5_kg_ha = p2o5,
  k2o_kg_ha = k2o,
  weed_control_sprays = weed_control_sprays,
  crop_rotation = crop_rotation,
  buds_count = buds_count
)

write_csv(df_sim, "cassava_survey.csv")
cat("Simulated cassava_survey.csv generated. Run analysis_pipeline.R next.\n")

