library(ggplot2)
library(ggpubr)
library(dplyr)
library(countreg)

show_dists <- function(df) {
  plots <- list()
  for(colname in colnames(df)) {
    
    if(class(df[,colname]) == "numeric" | class(df[,colname]) == "integer" ) {
      plot <- (ggplot(df) + geom_histogram(bins=30, aes_string(x = colname)))
      plots <- append(plots, list(plot))
    }
  }
  plots_per_row <- ifelse(length(plots)>=3, 3, length(plots))
  print(ggarrange(plotlist=plots, ncol = plots_per_row, nrow = ceiling(as.double(length(plots)) / as.double(plots_per_row))))
}

unify_missings <- function(df) {
  return (df |> mutate_all(~ ifelse(is.nan(.) | as.character(.) == "nan" | is.na(.), NA, .)))
}

na_rate <- function(df) {
  na_p <- t(df |> dplyr::summarise_all(list(~sum(is.na(.))/length(.))))
  NA_perctage_df <- data.frame (
    column_name = rownames(na_p),
    NA_percent = na_p
  )
  rownames(NA_perctage_df) <- NULL
  NA_perctage_df <- NA_perctage_df |> arrange(desc(NA_percent)) |> filter(NA_percent!=0)
  return (NA_perctage_df)
}

is_zero <- function(value) {
  return (value == 0)
}

impute_column_median <- function(df, column_name) {
  mid <- median(df[[column_name]], na.rm = TRUE)
  impute_column(df, column_name, mid)
}

impute_column <- function(df, column_name, new_value, bad_test = is.na, bad_indicator = TRUE) {
  if (bad_indicator) {
    # add is_bad as impute indicator
    impute_indicator_name <- paste("impute", column_name, sep = "_")
    df[impute_indicator_name] <- as.integer(bad_test(df[,column_name]))
  }
  
  # impute NA values
  df[column_name][bad_test(df[column_name])] <- new_value
  return (df)
}

case_summary <- function(column, name) {
  case_count <- summary(as.factor(column))
  data <- data.frame(
    case_name = names(case_count),
    case_count = case_count,
    case_ratio = case_count/length(column)
  )
  colnames(data)[1] <- paste(name, "case_name", sep="_")
  colnames(data)[2] <- paste(name, "case_count", sep="_")
  colnames(data)[3] <- paste(name, "case_ratio", sep="_")
  return (data)
}

fix_numeric_column <- function(df, column_name) {
  cuts <- unique(quantile(df[,column_name], probs=seq(0, 1, 0.1), na.rm=T))
  binned_column <- addNA(cut(df[[column_name]], cuts))
  levels(binned_column)[is.na(levels(binned_column))] <- "Unknown"
  df[paste("fix", column_name, sep="_")] <- binned_column
  return (df)
}

fix_categorical_column <- function(df, column_name) {
  column <- addNA(as.factor(df[[column_name]]))
  levels(column)[is.na(levels(column))] <- "Unknown"
  df[paste("fix", column_name, sep="_")] <- column
  return (df)
}

########################### Assess Model ##############################
# outCol: vector holding the values (known in the training step) of the
# output column that we want to predict, e.g., the 'churn'column.
# varCol: the single variable column that is of interest. Can we use this
# column alone to predict outCol?
# appCol: after building the model, we can apply it to this column (same
# as varCol but may come from the calibration or test set)
mkPredC <- function(outCol, varCol, appCol, pos = '1') {
  pPos <- sum(outCol == pos) / length(outCol)
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[pos]
  vTab <- table(as.factor(outCol), varCol)
  pPosWv <- (vTab[pos, ] + 1.0e-3*pPos) / (colSums(vTab) + 1.0e-3)
  pred <- pPosWv[appCol]
  pred[is.na(appCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
}

mkPredN <- function(outCol, varCol, appCol) {
  # compute the cuts
  cuts <- unique(
    quantile(varCol, probs=seq(0, 1, 0.1), na.rm=T))
  # discretize the numerical columns
  varC <- cut(varCol,cuts)
  appC <- cut(appCol,cuts)
  mkPredC(outCol,varC,appC)
}

library('ROCR')
calcAUC <- function(predcol, outcol, pos = '1') {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

calc_train_AUC <- function(df, column_name, target_name, pred_func) {
  pred <- pred_func(df[,target_name], df[,column_name], df[,column_name])
  aucTrain <- calcAUC(pred, df[,target_name])
  aucTrain
}

calc_test_AUC <- function(dfTrain, dfTest, column_name, target_name, pred_func) {
  pred <- pred_func(dfTrain[,target_name], dfTrain[,column_name], dfTest[,column_name])
  aucTest <- calcAUC(pred, dfTest[,target_name])
  aucTest
}

logLikelihood <- function(ytrue, ypred, epsilon=1e-6) {
  sum(ifelse(ytrue=='1', log(ypred+epsilon), log(1-ypred-epsilon)), na.rm=T)
}

calc_cross_validate_AUC <- function(df, column_name, target_name, pred_func) {
  aucs <- rep(0,100)
  for (rep in 1:length(aucs)) {
    useForCalRep <- rbinom(n=nrow(df), size=1, prob=0.1) > 0
    predRep <- pred_func(df[!useForCalRep, target_name],
                         df[!useForCalRep, column_name],
                         df[useForCalRep, column_name])
    aucs[rep] <- calcAUC(predRep, df[useForCalRep, target_name])
  }
  (aucs)
}

split.fun <- function(x, labs, digits, varlen, faclen)
{
  labs <- gsub(",", " ", labs)
  for(i in 1:length(labs)) {
    labs[i] <- paste(strwrap(labs[i], width=30), collapse="\n")
  }
  labs
}

assess_classifer <- function(true_values, pred_values) {
  rocobj <- roc(as.factor(true_values), pred_values)
  best_threshold <- coords(rocobj, "best")$threshold
  
  cm <- confusionMatrix(data= as.factor(as.numeric(pred_values > best_threshold)), 
                        reference = as.factor(true_values))
  
  tp <- cm$table[2, 2]
  tn <- cm$table[1, 1]
  fp <- cm$table[2, 1]
  fn <- cm$table[1, 2]
  accuracy <- (tp+tn) / (tp+tn+fp+fn)
  precision <- tp / (tp+fp)
  recall <- tp / (tp+fn)
  f1 <- 2 * precision * recall / (precision + recall)
  auc <- calcAUC(pred_values, true_values)
  c(accuracy, precision, recall, f1, auc)
}


print_clusters <- function(df, groups, cols_to_print) {
  Ngroups <- max(groups)
  for (i in 1:Ngroups) {
    print(paste("cluster", i))
    print(df[groups == i, cols_to_print])
  }
}

residual_plot <- function(model, title) {
  res <- residuals(model, type = "pearson")
  fit <-  fitted.values(model)
  data <- data.frame(
    res = res, 
    fit = fit
  )
  plt <- ggplot(data, aes(x = fit, y = res)) + geom_point() +ggtitle(title) + 
    theme(plot.title = element_text(size = 10)) + labs(x = "Fitted values", y = "Pearson Residuals")
  plt
  
  # plot(residuals(m1.poisson.simple, type = "pearson") ~ fitted.values(m1.poisson.simple),
  #      xlab = "Fitted values", ylab = "Pearson Residuals",
  #      main = "Poisson Model")
}


rooto_plot <- function(model, title) {
  plt <- autoplot(countreg::rootogram(model, plot = FALSE, style = "hanging"), 
                  colour = c("black", "darkblue"), 
                  size = c(1.2, 2)) + ggtitle(title)
  plt
}
