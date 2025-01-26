#============================================================================================================================
#======================Perfoming Imputations -DLGLM=================================================================================
#============================================================================================================================


# Environment setup
Sys.setenv(
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:64",
  RETICULATE_PYTHON="C:/Users/CHIKOMANA/anaconda3/envs/dlglm_env_new/python.exe",
  PYTORCH_NO_CUDA_MEMORY_CACHING="1",
  TORCH_WARN_ONCE="1"
)



reticulate::source_python("D:/Doctrate/chinese_guy_code/dlglm.py")


# Libraries and Python modules
library(reticulate)
library(data.table)

# Source required R files
source("D:/Doctrate/chinese_guy_code/dlglm.R")
source("D:/Doctrate/chinese_guy_code/prepareData.R")

# Initialize Python modules
np <- reticulate::import("numpy")
torch <- reticulate::import("torch")
warnings <- import("warnings")
warnings$filterwarnings('ignore')

# GPU settings
torch$cuda$set_per_process_memory_fraction(0.2)
torch$backends$cudnn$benchmark <- FALSE
torch$backends$cudnn$deterministic <- TRUE
torch$set_num_threads(as.integer(1))
torch$cuda$set_device(as.integer(0))

# Helper function for memory management
clean_memory <- function() {
  gc(reset = TRUE, full = TRUE)
  torch$cuda$empty_cache()
  torch$cuda$synchronize()
  Sys.sleep(10)
  invisible(gc())
}

# Function to calculate metrics
calculate_metrics <- function(result, missing_data, full_data, g, X_cols) {
  test_idx <- which(g == "test")
  mse_total <- 0
  mae_total <- 0
  total_count <- 0
  
  # Get test set imputations
  test_imputations <- as.data.frame(as.array(result$results_test$xhat))
  colnames(test_imputations) <- X_cols
  
  metrics_by_feature <- data.frame(
    feature = X_cols,
    mse = NA,
    rmse = NA,
    mae = NA,
    n_missing = NA
  )
  
  for(i in seq_along(X_cols)) {
    col <- X_cols[i]
    na_indices <- which(is.na(missing_data[[col]])[test_idx])
    
    if(length(na_indices) > 0) {
      true_vals <- full_data[[col]][test_idx[na_indices]]
      imputed_vals <- test_imputations[[col]][na_indices]
      
      # Calculate errors for this feature
      squared_errors <- (true_vals - imputed_vals)^2
      abs_errors <- abs(true_vals - imputed_vals)
      
      metrics_by_feature$mse[i] <- mean(squared_errors)
      metrics_by_feature$rmse[i] <- sqrt(metrics_by_feature$mse[i])
      metrics_by_feature$mae[i] <- mean(abs_errors)
      metrics_by_feature$n_missing[i] <- length(na_indices)
      
      # Accumulate total errors
      mse_total <- mse_total + sum(squared_errors)
      mae_total <- mae_total + sum(abs_errors)
      total_count <- total_count + length(na_indices)
    }
  }
  
  # Calculate overall metrics
  mse <- mse_total / total_count
  rmse <- sqrt(mse)
  mae <- mae_total / total_count
  
  return(list(
    overall = list(MSE = mse, RMSE = rmse, MAE = mae, Missing_Values = total_count),
    by_feature = metrics_by_feature
  ))
}

# Main imputation function
run_imputation <- function(mechanism) {
  base_dir <- "D:/Doctrate/all-three-missingness"
  missing_data_path <- file.path(base_dir, sprintf("missing_data_%s.csv", tolower(mechanism)))
  full_data_path <- file.path(base_dir, "full_data.csv")
  
  # Load data
  missing_data <- fread(missing_data_path)
  full_data <- fread(full_data_path)
  
  # Prepare data
  X_cols <- paste0("X", 1:25)
  X <- as.matrix(missing_data[, ..X_cols])
  Y <- as.matrix(missing_data[, .(Y)])
  
  # Calculate dimensions
  P <- ncol(X)
  N <- nrow(X)
  
  # Create splits
  set.seed(123)
  g <- sample(c("train", "valid", "test"),
              size = N,
              replace = TRUE,
              prob = c(0.8, 0.1, 0.1))
  
  # Create masks
  mask_x <- 1 * (!is.na(X))
  mask_y <- 1 * (!is.na(Y))
  
  # Define hyperparameters exactly as in source
  hyperparams <- list(
    sigma = "elu",
    bss = c(1000L),
    lrs = c(0.01, 0.001),
    impute_bs = 1000L,
    arch = "IWAE",
    niws = 5L,
    n_imps = 500L,
    n_epochss = 2002L,
    n_hidden_layers = c(0L, 1L, 2L),
    n_hidden_layers_y = c(0L),
    n_hidden_layers_r = c(0L),  # Set to 0 for all mechanisms as per source
    h = c(128L, 64L),
    h_y = NULL,
    h_r = c(16L, 32L),
    dim_zs = c(as.integer(floor(P/12)),
               as.integer(floor(P/4)),
               as.integer(floor(P/2)),
               as.integer(floor(3*P/4))),
    L1_weights = if(mechanism == "MNAR") c(1e-1, 5e-2, 0) else 0
  )
  
  # Run DLGLM
  result <- dlglm(
    dir_name = file.path(base_dir, "results", mechanism),
    X = X,
    Y = Y,
    mask_x = mask_x,
    mask_y = mask_y,
    g = g,
    covars_r_x = rep(1, ncol(X)),
    covars_r_y = 1,
    learn_r = TRUE,
    data_types_x = rep("real", ncol(X)),
    Ignorable = mechanism %in% c("MCAR", "MAR"),
    family = "Multinomial",
    link = "mlogit",
    normalize = TRUE,
    early_stop = TRUE,
    trace = TRUE,
    draw_miss = TRUE,
    init_r = if(mechanism == "MNAR") "alt" else "default",  # Use alternative initialization for MNAR
    hyperparams = hyperparams
  )
  
  # Calculate and save metrics
  if (!is.null(result)) {
    metrics <- calculate_metrics(result, missing_data, full_data, g, X_cols)
    
    # Save detailed metrics
    results_dir <- file.path(base_dir, "results", mechanism)
    dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Save overall metrics
    metrics_df <- data.frame(
      Mechanism = mechanism,
      MSE = metrics$overall$MSE,
      RMSE = metrics$overall$RMSE,
      MAE = metrics$overall$MAE,
      Missing_Values = metrics$overall$Missing_Values
    )
    
    # Save results
    fwrite(metrics_df, file.path(results_dir, "overall_metrics.csv"))
    fwrite(metrics$by_feature, file.path(results_dir, "feature_metrics.csv"))
    
    # Print current metrics
    cat("\nMetrics for", mechanism, ":\n")
    print(metrics_df)
    
    return(metrics_df)
  }
  
  return(NULL)
}

# Run for each mechanism
mechanisms <- c("MAR", "MNAR", "MCAR")
all_metrics <- list()

for(mech in mechanisms) {
  cat("\nProcessing", mech, "...\n")
  clean_memory()  # Clear memory before each run
  metrics <- run_imputation(mech)
  all_metrics[[mech]] <- metrics
}

# Combine and save final metrics
final_metrics <- do.call(rbind, all_metrics)
fwrite(final_metrics, "D:/Doctrate/all-three-missingness/final_metrics.csv")
print("\nFinal metrics for all mechanisms:")
print(final_metrics)
