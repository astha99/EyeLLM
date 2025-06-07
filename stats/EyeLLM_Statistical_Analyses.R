library(dplyr)
library(tidyr)
library(lme4)
library(stringr)
library(emmeans)
library(readr)
library(lmerTest)  # for p-values

# List of file names
files <- c("gpt3.5.csv", "gpt4.csv", "mistral.csv", "llama3.csv")

LLMs <- sub("\\.csv$", "", files)
# Read each file and add a 'llm' column
read_and_label <- function(file) {
  read_csv(file.path("data", file)) %>%
    mutate(llm = gsub("\\.csv", "", file))
}

# Read and combine all files into one dataframe
df <- bind_rows(lapply(files, read_and_label))

df <- df %>%
  mutate(x_example_id = paste0(token_id, as.character(example_id))) %>% 
  mutate(condition = recode(condition,
                            "Pretext + Sentences" = "Sentences",
                            "Pretext + Words" = "Words",
                            "Pretext Only" = "Baseline",
                            "Control" = "Control"))


run_analysis_across_all_models <- function(data, measure_pattern) {
  # Filter dataset based on the condition pattern
  subset <- data %>% filter(str_detect(measure, measure_pattern))
  
  subset$condition <- factor(subset$condition)
  subset$condition <- relevel(subset$condition, ref = "Control")
  

  # Fit the mixed-effects models
  model0 <- lmer(similarity_score ~ 1 + (1 | x_example_id), data = subset)
  model1 <- lmer(similarity_score ~ 0 + condition + (1 | x_example_id), data = subset)
  model2 <- lmer(similarity_score ~ 0 + condition + llm + (1 | x_example_id), data = subset)
  model3 <- lmer(similarity_score ~ 0 + llm + condition + llm:condition + (1 | x_example_id), data = subset)
  
  # chi-squared test for comparison of models (M0 vs. M1, M1 vs. M2, M2 vs. M3)
  anova_result_01 <- anova(model0, model1)
  anova_result_12 <- anova(model1, model2)
  anova_result_23 <- anova(model2, model3)

  # Obtain estimated marginal means
  # Do pairwise comparisons with Tukey adjustment
  
  # LLM first (for RQ1)
  pairwise_comparisons <- pairs(emmeans(model3, ~llm), adjust = "tukey")
  
  # by Condition (for RQ2)
  pairwise_comparisons_2 <- pairs(emmeans(model3, ~condition), adjust = "tukey")
  
  # Return results as a list
  return(list(
    anova_01 = anova_result_01,
    anova_12 = anova_result_12,
    anova_23 = anova_result_23,
    pairwise_comparisons = pairwise_comparisons,
    pairwise_comparisons_2 = pairwise_comparisons_2
  ))
}


# by-LLM analyses

run_analysis <- function(data, measure_pattern) {
  # Filter for measure == 'Semantic'
  filtered_df <- data %>% filter(measure == measure_pattern)
  
  # Convert condition to factor and set Baseline as reference
  filtered_df$condition <- factor(filtered_df$condition)
  filtered_df$condition <- relevel(filtered_df$condition, ref = "Control")
  
  # Fit the model
  model0 <- lmer(similarity_score ~ 1 + (1 + condition | x_example_id), data = filtered_df)
  model <- lmer(similarity_score ~ condition + (1 + condition | x_example_id), data = filtered_df)

  cat("\n### Improvement of model fit M0 vs. M1 ###\n")

  print(anova(model0,model))
  
  cat("\n### Significance of coefficients in M1 ###\n")

  print(summary(model))
  
  #plot(fitted(model), resid(model))
  #abline(h = 0, col = "red")
  
  #hist(resid(model), breaks = 30)
  #qqnorm(resid(model))
  #qqline(resid(model), col = "red")
}



cat("\n\nRQ1: Comparisons by LLM\n\n");


for (iter_llm in LLMs) {
  df_llm <- subset(df, llm == iter_llm)

  cat("\n\n\n** LLM = ", toupper(iter_llm), "\n\n")

  cat("\n\n\n*** Metric: SEMANTIC\n\n") 
  run_analysis(df_llm, "Semantic")

  cat("\n\n\n*** Metric: JACCARD\n\n") 
  run_analysis(df_llm, "Jaccard")

  cat("\n\n\n*** Metric: F1\n\n") 
  run_analysis(df_llm, "F1")
}



cat("\n\nRQ2: Comparisons across all LLMs\n\n");

cat("\n\n** Metric: SEMANTIC\n\n")
res <- run_analysis_across_all_models(df, "Semantic")
print(res)


cat("\n\n** Metric: JACCARD\n\n")
res <- run_analysis_across_all_models(df, "Jaccard")
print(res)

cat("\n\n** Metric: F1\n\n")
res <- run_analysis_across_all_models(df, "F1")
print(res)

