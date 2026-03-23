# MATH42815 Machine Learning - Assignment 2
# Wine Quality Prediction: Elastic Net vs Random Forest
# Data: UCI Wine Quality (Vinho Verde, Portugal)

# ---- 0. Setup ----

required_packages <- c("tidyverse", "caret", "glmnet", "randomForest",
                       "RColorBrewer", "corrplot")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

set.seed(42)

# colour/shape settings for plots
cat_palette  <- brewer.pal(3, "Set1")
wine_colours <- c("red" = cat_palette[1], "white" = cat_palette[2])
wine_shapes  <- c("red" = 16, "white" = 17)   # circle vs triangle

theme_set(theme_minimal(base_size = 12) +
            theme(plot.title   = element_text(size = 13, face = "bold"),
                  axis.title   = element_text(size = 11),
                  legend.title = element_text(size = 10)))

# ---- 1. Load and merge data ----

red   <- read.csv("data/winequality-red.csv",   sep = ";", header = TRUE)
white <- read.csv("data/winequality-white.csv", sep = ";", header = TRUE)

red$type   <- "red"
white$type <- "white"
wine       <- rbind(red, white)
wine$type  <- as.factor(wine$type)

names(wine) <- gsub("\\.", "_", names(wine))

dim(wine)              # 6497 x 13
table(wine$type)

# ---- 2. Data cleaning ----

str(wine)

# missing values
colSums(is.na(wine))   # all zero

# duplicates
n_dup <- sum(duplicated(wine))
n_dup   # 1177
wine <- wine[!duplicated(wine), ]
nrow(wine)   # 5320

summary(wine)

# quality distribution
table(wine$quality)
mean(wine$quality)     # 5.80
median(wine$quality)   # 6

# ---- 3. Exploratory Data Analysis ----

# Fig 1: quality distribution by type
fig1 <- ggplot(wine, aes(x = factor(quality), fill = type)) +
  geom_bar(position = "dodge", alpha = 0.85) +
  scale_fill_manual(values = wine_colours, name = "Wine Type") +
  labs(title = "Figure 1: Distribution of Wine Quality Scores by Type",
       x = "Quality Score", y = "Count") +
  facet_wrap(~ type, scales = "free_y") +
  theme(legend.position = "bottom")

print(fig1)
ggsave("fig1_quality_distribution.png", fig1, width = 8, height = 5, dpi = 300)

# Fig 2: correlation heatmap
numeric_vars <- wine %>% select(where(is.numeric))
cor_matrix   <- cor(numeric_vars, use = "complete.obs")

png("fig2_correlation_heatmap.png", width = 2400, height = 2000, res = 300)
corrplot(cor_matrix,
         method = "color", type = "upper", order = "hclust",
         col = brewer.pal(11, "RdYlBu"),
         tl.col = "black", tl.cex = 0.8,
         addCoef.col = "black", number.cex = 0.6,
         title = "Figure 2: Correlation Heatmap of Wine Features",
         mar = c(0, 0, 2, 0))
dev.off()

# Fig 3: key features vs quality (boxplots)
key_features <- c("alcohol", "volatile_acidity", "sulphates", "citric_acid")

fig3_data <- wine %>%
  select(all_of(key_features), quality) %>%
  pivot_longer(cols = all_of(key_features),
               names_to = "feature", values_to = "value")

fig3 <- ggplot(fig3_data, aes(x = factor(quality), y = value,
                               fill = factor(quality))) +
  geom_boxplot(alpha = 0.7, outlier.size = 0.8) +
  scale_fill_brewer(palette = "RdYlBu", name = "Quality") +
  facet_wrap(~ feature, scales = "free_y", ncol = 2) +
  labs(title = "Figure 3: Distribution of Key Features Across Quality Scores",
       x = "Quality Score", y = "Value") +
  theme(legend.position = "none")

print(fig3)
ggsave("fig3_features_vs_quality.png", fig3, width = 10, height = 8, dpi = 300)

# Fig 4: alcohol vs density scatter plot
fig4 <- ggplot(wine, aes(x = alcohol, y = density,
                          colour = type, shape = type)) +
  geom_point(alpha = 0.4, size = 1.5) +
  scale_colour_manual(values = wine_colours, name = "Wine Type") +
  scale_shape_manual(values = wine_shapes, name = "Wine Type") +
  labs(title = "Figure 4: Alcohol Content vs Density by Wine Type",
       x = "Alcohol (% vol)", y = "Density (g/cm\u00B3)") +
  theme(legend.position = "bottom")

print(fig4)
ggsave("fig4_alcohol_vs_density.png", fig4, width = 8, height = 6, dpi = 300)

# Fig 5: density plots by wine type
features_to_plot <- c("alcohol", "volatile_acidity", "residual_sugar",
                      "total_sulfur_dioxide", "density", "pH")

fig5_data <- wine %>%
  select(all_of(features_to_plot), type) %>%
  pivot_longer(cols = all_of(features_to_plot),
               names_to = "feature", values_to = "value")

fig5 <- ggplot(fig5_data, aes(x = value, fill = type, colour = type)) +
  geom_density(alpha = 0.4) +
  scale_fill_manual(values = wine_colours, name = "Wine Type") +
  scale_colour_manual(values = wine_colours, name = "Wine Type") +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  labs(title = "Figure 5: Feature Distributions by Wine Type",
       x = "Value", y = "Density") +
  theme(legend.position = "bottom")

print(fig5)
ggsave("fig5_density_plots_by_type.png", fig5, width = 12, height = 8, dpi = 300)

# Fig 6: pairs plot
key_vars <- c("alcohol", "volatile_acidity", "sulphates",
              "citric_acid", "quality")

png("fig6_pairs_plot.png", width = 2400, height = 2400, res = 300)
pairs(wine[, key_vars],
      col = ifelse(wine$type == "red", wine_colours["red"], wine_colours["white"]),
      pch = ifelse(wine$type == "red", 16, 17),
      cex = 0.4,
      main = "Figure 6: Pairwise Scatter Plot of Key Features",
      lower.panel = NULL)
par(xpd = TRUE)
legend("bottomleft", legend = c("Red", "White"),
       col = wine_colours, pch = c(16, 17), cex = 0.9, bty = "n")
dev.off()

# ---- 4. Train/test split and preprocessing ----

# stratified 80/20 split
set.seed(42)
train_index <- createDataPartition(wine$quality, p = 0.8, list = FALSE)
train_data  <- wine[train_index, ]
test_data   <- wine[-train_index, ]

nrow(train_data)   # 4258
nrow(test_data)    # 1062

# scale numeric predictors for Elastic Net (using training-set stats only)
numeric_predictors <- setdiff(names(wine)[sapply(wine, is.numeric)], "quality")

pre_process_params <- preProcess(train_data[, numeric_predictors],
                                 method = c("center", "scale"))

train_scaled <- predict(pre_process_params, train_data)
test_scaled  <- predict(pre_process_params, test_data)

# model matrices for glmnet
x_train <- model.matrix(quality ~ ., data = train_scaled)[, -1]
y_train <- train_scaled$quality
x_test  <- model.matrix(quality ~ ., data = test_scaled)[, -1]
y_test  <- test_scaled$quality

# shared 10-fold CV folds for both models
set.seed(42)
cv_folds    <- createFolds(train_data$quality, k = 10, returnTrain = TRUE)
shared_ctrl <- trainControl(method = "cv", number = 10, index = cv_folds)

# ---- 5. Elastic Net ----

# tune alpha over [0, 1]
alpha_values  <- seq(0, 1, by = 0.1)
alpha_results <- data.frame(alpha = alpha_values, min_lambda = NA, cv_rmse = NA)

set.seed(42)
for (i in seq_along(alpha_values)) {
  cv_fit <- cv.glmnet(x_train, y_train,
                      alpha = alpha_values[i],
                      nfolds = 10,
                      type.measure = "mse")
  alpha_results$min_lambda[i] <- cv_fit$lambda.min
  alpha_results$cv_rmse[i]    <- sqrt(min(cv_fit$cvm))
}

alpha_results

best_alpha  <- alpha_results$alpha[which.min(alpha_results$cv_rmse)]
best_lambda <- alpha_results$min_lambda[which.min(alpha_results$cv_rmse)]
best_alpha    # 0.1
best_lambda

# Fig 7: alpha tuning
fig7 <- ggplot(alpha_results, aes(x = alpha, y = cv_rmse)) +
  geom_line(colour = cat_palette[1], linewidth = 1) +
  geom_point(colour = cat_palette[1], size = 3) +
  geom_vline(xintercept = best_alpha, linetype = "dashed", colour = "grey40") +
  annotate("text", x = best_alpha + 0.05, y = max(alpha_results$cv_rmse),
           label = paste("Best alpha =", best_alpha), hjust = 0, size = 3.5) +
  labs(title = "Figure 7: Elastic Net - Alpha Tuning via Cross-Validation",
       x = "Alpha (L1/L2 Mixing Parameter)", y = "Cross-Validated RMSE")

print(fig7)
ggsave("fig7_alpha_tuning.png", fig7, width = 8, height = 5, dpi = 300)

# fit cv.glmnet at best alpha for the lambda path plot
set.seed(42)
enet_cv <- cv.glmnet(x_train, y_train, alpha = best_alpha,
                     nfolds = 10, type.measure = "mse")

# Fig 8: lambda selection
png("fig8_lambda_path.png", width = 2400, height = 1800, res = 300)
par(mar = c(5, 4, 6, 2))
plot(enet_cv, main = "")
title("Figure 8: Elastic Net - Lambda Selection via CV", line = 4.5)
dev.off()

# final model
enet_final <- glmnet(x_train, y_train,
                     alpha = best_alpha, lambda = enet_cv$lambda.min)

# Fig 9: coefficients
enet_coefs <- as.matrix(coef(enet_final))
enet_coefs_df <- data.frame(
  feature     = rownames(enet_coefs),
  coefficient = enet_coefs[, 1]
)
enet_coefs_df <- enet_coefs_df[enet_coefs_df$feature != "(Intercept)", ]
enet_coefs_df <- enet_coefs_df[order(abs(enet_coefs_df$coefficient),
                                     decreasing = TRUE), ]
enet_coefs_df$feature <- factor(enet_coefs_df$feature,
                                levels = enet_coefs_df$feature)

fig9 <- ggplot(enet_coefs_df, aes(x = feature, y = coefficient,
                                   fill = coefficient > 0)) +
  geom_col(alpha = 0.85) +
  scale_fill_manual(values = c(cat_palette[1], cat_palette[2]),
                    labels = c("Negative", "Positive"), name = "Direction") +
  coord_flip() +
  labs(title = "Figure 9: Elastic Net - Feature Coefficients",
       subtitle = paste("Alpha =", best_alpha, ", Lambda =",
                        round(best_lambda, 5)),
       x = "Feature", y = "Coefficient Value") +
  theme(legend.position = "bottom")

print(fig9)
ggsave("fig9_enet_coefficients.png", fig9, width = 8, height = 6, dpi = 300)

# test set performance
enet_pred <- as.vector(predict(enet_final, newx = x_test,
                               s = enet_cv$lambda.min))

enet_rmse <- sqrt(mean((enet_pred - test_data$quality)^2))
enet_mae  <- mean(abs(enet_pred - test_data$quality))
enet_r2   <- 1 - sum((test_data$quality - enet_pred)^2) /
                 sum((test_data$quality - mean(test_data$quality))^2)

round(c(RMSE = enet_rmse, MAE = enet_mae, R2 = enet_r2), 4)

# ---- 6. Random Forest ----

# tune mtry with shared CV folds
set.seed(42)
rf_model <- train(
  quality ~ ., data = train_data,
  method    = "rf",
  trControl = shared_ctrl,
  tuneGrid  = expand.grid(mtry = 2:12),
  ntree     = 100,
  importance = TRUE,
  metric    = "RMSE"
)

rf_model$results
rf_model$bestTune$mtry   # 3

# Fig 10: mtry tuning
fig10 <- ggplot(rf_model$results, aes(x = mtry, y = RMSE)) +
  geom_line(colour = cat_palette[2], linewidth = 1) +
  geom_point(colour = cat_palette[2], size = 3) +
  geom_errorbar(aes(ymin = RMSE - RMSESD, ymax = RMSE + RMSESD),
                width = 0.3, colour = "grey50") +
  geom_vline(xintercept = rf_model$bestTune$mtry,
             linetype = "dashed", colour = "grey40") +
  annotate("text", x = rf_model$bestTune$mtry + 0.3,
           y = max(rf_model$results$RMSE),
           label = paste("Best mtry =", rf_model$bestTune$mtry),
           hjust = 0, size = 3.5) +
  labs(title = "Figure 10: Random Forest - mtry Tuning via Cross-Validation",
       x = "mtry (Number of Features at Each Split)",
       y = "Cross-Validated RMSE (\u00B11 SD)")

print(fig10)
ggsave("fig10_rf_mtry_tuning.png", fig10, width = 8, height = 5, dpi = 300)

# examine ntree convergence
best_mtry    <- rf_model$bestTune$mtry
ntree_values <- c(100, 200, 300, 500, 700, 1000)
ntree_results <- data.frame(ntree = ntree_values, OOB_RMSE = NA)

set.seed(42)
for (i in seq_along(ntree_values)) {
  rf_temp <- randomForest(quality ~ ., data = train_data,
                          mtry = best_mtry, ntree = ntree_values[i],
                          importance = FALSE)
  ntree_results$OOB_RMSE[i] <- sqrt(tail(rf_temp$mse, 1))
}

ntree_results

# Fig 11: ntree convergence
fig11 <- ggplot(ntree_results, aes(x = ntree, y = OOB_RMSE)) +
  geom_line(colour = cat_palette[2], linewidth = 1) +
  geom_point(colour = cat_palette[2], size = 3) +
  labs(title = "Figure 11: Random Forest - OOB Error Convergence",
       x = "Number of Trees (ntree)", y = "Out-of-Bag RMSE")

print(fig11)
ggsave("fig11_rf_ntree_convergence.png", fig11, width = 8, height = 5, dpi = 300)

# final model with 500 trees
set.seed(42)
rf_final <- randomForest(quality ~ ., data = train_data,
                         mtry = best_mtry, ntree = 500, importance = TRUE)

# Fig 12: variable importance
importance_df <- as.data.frame(importance(rf_final))
importance_df$feature <- rownames(importance_df)
importance_df <- importance_df[order(importance_df$`%IncMSE`, decreasing = TRUE), ]
importance_df$feature <- factor(importance_df$feature,
                                levels = rev(importance_df$feature))

fig12 <- ggplot(importance_df, aes(x = feature, y = `%IncMSE`)) +
  geom_col(fill = cat_palette[2], alpha = 0.85) +
  coord_flip() +
  labs(title = "Figure 12: Random Forest - Variable Importance",
       subtitle = paste("mtry =", best_mtry, ", ntree = 500"),
       x = "Feature", y = "% Increase in MSE When Permuted")

print(fig12)
ggsave("fig12_rf_variable_importance.png", fig12, width = 8, height = 6, dpi = 300)

# test set performance
rf_pred <- predict(rf_final, newdata = test_data)

rf_rmse <- sqrt(mean((rf_pred - test_data$quality)^2))
rf_mae  <- mean(abs(rf_pred - test_data$quality))
rf_r2   <- 1 - sum((test_data$quality - rf_pred)^2) /
               sum((test_data$quality - mean(test_data$quality))^2)

round(c(RMSE = rf_rmse, MAE = rf_mae, R2 = rf_r2), 4)

# ---- 7. Model comparison ----

# train Elastic Net through caret so we can use resamples()
set.seed(42)
enet_caret <- train(
  quality ~ ., data = train_scaled,
  method    = "glmnet",
  trControl = shared_ctrl,
  tuneGrid  = expand.grid(alpha = best_alpha, lambda = enet_cv$lambda.min),
  metric    = "RMSE"
)

model_list     <- list(ElasticNet = enet_caret, RandomForest = rf_model)
resamp_results <- resamples(model_list)
summary(resamp_results)

# Fig 13: CV comparison boxplot
png("fig13_cv_comparison.png", width = 2400, height = 1600, res = 300)
bwplot(resamp_results, metric = "RMSE",
       main = "Figure 13: Cross-Validated RMSE Comparison")
dev.off()

# test set summary table
comparison <- data.frame(
  Model = c("Elastic Net", "Random Forest"),
  RMSE  = round(c(enet_rmse, rf_rmse), 4),
  MAE   = round(c(enet_mae, rf_mae), 4),
  R2    = round(c(enet_r2, rf_r2), 4)
)
comparison

# Fig 14: predicted vs actual
pred_df <- data.frame(
  Actual    = rep(test_data$quality, 2),
  Predicted = c(enet_pred, rf_pred),
  Model     = rep(c("Elastic Net", "Random Forest"), each = nrow(test_data))
)

fig14 <- ggplot(pred_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.3, colour = cat_palette[2], size = 1.2) +
  geom_abline(intercept = 0, slope = 1,
              linetype = "dashed", colour = cat_palette[1], linewidth = 0.8) +
  facet_wrap(~ Model) +
  labs(title = "Figure 14: Predicted vs Actual Wine Quality",
       x = "Actual Quality", y = "Predicted Quality") +
  coord_equal(xlim = c(3, 9), ylim = c(3, 9))

print(fig14)
ggsave("fig14_predicted_vs_actual.png", fig14, width = 10, height = 5, dpi = 300)

# Fig 15: residuals
resid_df <- data.frame(
  Fitted   = c(enet_pred, rf_pred),
  Residual = c(test_data$quality - enet_pred, test_data$quality - rf_pred),
  Model    = rep(c("Elastic Net", "Random Forest"), each = nrow(test_data))
)

fig15 <- ggplot(resid_df, aes(x = Fitted, y = Residual)) +
  geom_point(alpha = 0.3, colour = "grey40", size = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = cat_palette[1]) +
  geom_smooth(method = "loess", se = FALSE,
              colour = cat_palette[2], linewidth = 0.8) +
  facet_wrap(~ Model) +
  labs(title = "Figure 15: Residual Analysis",
       x = "Fitted Value", y = "Residual (Actual - Predicted)")

print(fig15)
ggsave("fig15_residual_analysis.png", fig15, width = 10, height = 5, dpi = 300)

# Fig 16: variable importance comparison (normalised to 0-100)
enet_imp <- data.frame(
  feature    = enet_coefs_df$feature,
  importance = abs(enet_coefs_df$coefficient),
  Model      = "Elastic Net"
)
enet_imp$importance <- 100 * enet_imp$importance / max(enet_imp$importance)

rf_imp <- data.frame(
  feature    = importance_df$feature,
  importance = importance_df$`%IncMSE`,
  Model      = "Random Forest"
)
rf_imp$importance <- 100 * rf_imp$importance / max(rf_imp$importance)

combined_imp <- rbind(enet_imp, rf_imp)

fig16 <- ggplot(combined_imp, aes(x = feature, y = importance, fill = Model)) +
  geom_col(position = "dodge", alpha = 0.85) +
  scale_fill_manual(values = c("Elastic Net" = cat_palette[1],
                               "Random Forest" = cat_palette[2])) +
  coord_flip() +
  labs(title = "Figure 16: Variable Importance Comparison",
       x = "Feature", y = "Relative Importance (%)") +
  theme(legend.position = "bottom")

print(fig16)
ggsave("fig16_variable_importance_comparison.png", fig16,
       width = 10, height = 6, dpi = 300)

# ---- 8. Summary ----

comparison
# Random Forest wins on all three metrics (lower RMSE/MAE, higher R2).
# Both models agree that alcohol and volatile acidity are the top predictors.
