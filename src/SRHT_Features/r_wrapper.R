# run_isrht.R

# --- 1. Robust Package Installation ---
ensure_package <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    if (!require(pkg, character.only = TRUE)) stop(paste("Failed to install", pkg))
  }
}
ensure_package("Rcpp")
ensure_package("RcppEigen")
ensure_package("data.table")

library(Rcpp)
library(RcppEigen)
library(data.table)

# --- CONFIGURATION ---
sample_size_r <- 50
bin_count <- 10

# --- 2. Compile ---
message("Compiling C++ source...")
sourceCpp("Feature_selection.cpp") 

# --- 3. Print Helper (SHOWS ALL COEFFICIENTS) ---
print_results <- function(res_list, title_str) {
  message(paste0("\n=========================================="))
  message(paste0(" RESULTS: ", title_str))
  message(paste0("=========================================="))
  
  if (length(res_list) == 0) return()
  
  for (item in res_list) {
    method <- item$Method
    r2 <- round(item$R_Squared, 4)
    raw_coefs <- item$Coefficients
    selected_indices <- item$Indices + 1 
    total_d <- item$TotalFeatures
    
    cat(sprintf("[Method: %-10s] -> R^2: %.4f\n", method, r2))
    
    if (all(raw_coefs == 0) && length(raw_coefs) > 1) {
      cat("  WARNING: Coefficients are all 0. Regression likely failed.\n\n")
      next
    }
    
    full_vector <- numeric(total_d)
    intercept <- raw_coefs[1]
    weights <- raw_coefs[-1]
    
    if(length(weights) == length(selected_indices)) {
      full_vector[selected_indices] <- weights
    }
    
    cat(sprintf("  Intercept: %.4f\n", intercept))
    cat(sprintf("  Full Feature Vector (All %d values): \n  ", total_d))
    cat(round(full_vector, 4), fill = TRUE)
    cat("\n")
  }
}

# --- 4. Wrapper Function ---
run_isrht_wrapper <- function(user_csv_path = NULL, run_synthetic = TRUE) {
  
  # --- Scenario A: Synthetic Data ---
  if (run_synthetic) {
    message(paste0("\n--- Running on Synthetic Data (r=", sample_size_r, ") ---"))
    set.seed(42)
    N <- 2000; d <- 500
    X_syn <- matrix(rnorm(N * d), nrow = N, ncol = d)
    true_w <- rnorm(d)
    y_syn <- X_syn %*% true_w + rnorm(N, 0, 0.5) 
    
    results_syn <- run_isrht_benchmark(X_syn, as.vector(y_syn), sample_size_r, bin_count)
    print_results(results_syn, "Synthetic Data")
  }
  
  # --- Scenario B: User CSV ---
  if (!is.null(user_csv_path) && file.exists(user_csv_path)) {
    message(paste0("\n--- Running on User CSV (r=", sample_size_r, ") ---"))
    
    data <- data.table::fread(user_csv_path)
    y_col_idx <- ncol(data)
    X_user <- as.matrix(data[, -..y_col_idx])
    y_user <- as.vector(data[[y_col_idx]])
    
    # Data Cleaning for Stability
    X_user[is.na(X_user)] <- 0
    if(any(is.na(y_user))) y_user[is.na(y_user)] <- mean(y_user, na.rm=TRUE)
    
    if (ncol(X_user) < sample_size_r) sample_size_r <- ncol(X_user)
    
    results_user <- run_isrht_benchmark(X_user, y_user, sample_size_r, bin_count)
    print_results(results_user, "User CSV Data")
  }
}

# --- Execute ---
run_isrht_wrapper()
run_isrht_wrapper("chem_data.csv", run_synthetic = FALSE)
