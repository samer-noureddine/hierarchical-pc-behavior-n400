library(lme4)
library(lmerTest)
library(tidyverse)
library(emmeans)
library(effectsize)


# prevent R from wrapping its output around too soon
options(width=300)

base = "C:/Users/samer/postdoc_projects/~hierarchicalPC/comprehension/PC_KLdiv_multiplicative_PCBCDIM/simulation_csv_files//"

# load data
reppriming_sim= read_csv(paste(base, "RepetitionPriming_Simulation_N400_27_to_31_IterationsToThreshold2.8.csv", sep = ""))
sempriming_sim = read_csv(paste(base, "SemanticPriming_Simulation_N400_27_to_31_IterationsToThreshold2.8.csv", sep = ""))
cloze_sim = read_csv(paste(base, "ClozeProbability_Simulation_N400_22_to_26_IterationsToThreshold2.8.csv", sep = ""))
lexviol_sim = read_csv(paste(base, "LexicalViolation_Simulation_N400_22_to_26_IterationsToThreshold2.8.csv", sep = ""))
sempredoverlap_sim= read_csv(paste(base, "SemanticPredictionOverlap_Simulation_N400_22_to_26_IterationsToThreshold2.8.csv", sep = ""))
competition_sim= read_csv(paste(base, "Competition_Simulation_N400_22_to_26_IterationsToThreshold2.8.csv", sep = ""))

numstims = reppriming_sim %>% select(WordInds) %>% unique() %>% summarize(n = n()) %>% pull(n)


###### Simulation 1 and 2: WORD PAIR PRIMING #######

options(width = 100)

reppriming_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(Repeated_code) %>% summarize(accuracy = sum(CorrectItemCrossed)/n())

reppriming_sim_accurate = reppriming_sim %>% filter(CorrectItemCrossed == 1)
# reppriming_sim_accurate %>% group_by(Repeated_name) %>% summarize(n=n())
# failed to converge:
RepPrimModel = lmer(LexSemErr ~ Repeated_code + (1|WordInput) + (0+ Repeated_code| WordInput), data = reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
RepPrimModel_N400 = lmer(LexSemErr ~ Repeated_code + (1 | WordInput), data = reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(RepPrimModel_N400)
# failed to converge:
RepPrimModel_behav = lmer(ThresholdCrossingIteration ~ Repeated_code + (1 | WordInput) + (0 + Repeated_code ||WordInput), data = reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
# RepPrimModel_behav = lmer(ThresholdCrossingIteration ~ Repeated_code + (1 | WordInput), data = reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(RepPrimModel_behav)


# Semantic priming simulation
# sempriming_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(SemanticRelatedness_code) %>% summarize(accuracy = sum(CorrectItemCrossed)/n())
sempriming_sim_accurate = sempriming_sim %>% filter(CorrectItemCrossed == 1)
# sempriming_sim_accurate %>% group_by(SemanticRelatedness_name) %>% summarize(n=n()/numstims)
# failed to converge:
SemPrimModel_N400= lmer(LexSemErr ~ SemanticRelatedness_code + (1 | WordInput)+ (0 +SemanticRelatedness_code|WordInput ), data = sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
SemPrimModel_N400= lmer(LexSemErr ~ SemanticRelatedness_code + (1 | WordInput), data = sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SemPrimModel_N400)
SemPrimModel_behav= lmer(ThresholdCrossingIteration ~ SemanticRelatedness_code + (1 | WordInput)+ (0 +SemanticRelatedness_code|WordInput ), data = sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SemPrimModel_behav)

# difference between repeated and semantically primed words
repprimed = reppriming_sim_accurate %>% filter(Repeated_name == "repeated")%>% mutate(dummy_condition = "repeated")
semprimed = sempriming_sim_accurate %>% filter(SemanticRelatedness_name == "semrelated") %>% mutate(dummy_condition = "semrelated")
sem_vs_rep = full_join(semprimed, repprimed, by = "WordInds")
t.test(sem_vs_rep$LexSemErr.x, sem_vs_rep$LexSemErr.y, paired = TRUE)
t.test(sem_vs_rep$ThresholdCrossingIteration.x, sem_vs_rep$ThresholdCrossingIteration.y, paired = TRUE)

###### Simulation 3: LEXICAL PROBABILITY #######

# Top-down preactivation simulations
# cloze_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(as.factor(Cloze_code)) %>% summarize(accuracy = sum(CorrectItemCrossed)/n(), n = n())
cloze_sim_accurate = cloze_sim %>% filter(CorrectItemCrossed == 1)
# cloze_sim_accurate %>% group_by(Cloze_code) %>% summarize(n=n()/numstims)
ClzModel_N400 = lmer(LexSemErr ~ scale(Cloze_code) + (scale(Cloze_code)  | WordInput), data = cloze_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(ClzModel_N400)
ClzModel_behav = lmer(ThresholdCrossingIteration ~ scale(Cloze_code) + (scale(Cloze_code)  | WordInput), data = cloze_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(ClzModel_behav)

###### Simulations 4 and 5: NO PROCESSING COSTS FOR PREDICTION VIOLATION #######

###### Simulation 4: Lexical prediction violation
lexviol_sim_accurate = lexviol_sim %>% filter(CorrectItemCrossed == 1)
lexviol_sim_accurate$ExpConstr_Condition <- factor(
  paste(lexviol_sim_accurate$Constraint_name, lexviol_sim_accurate$IsExpected_name, sep = "_")
)
lexviol_sim_accurate$ExpConstr_Condition = relevel(lexviol_sim_accurate$ExpConstr_Condition,ref = "LowConstraint_Unexpected")
# failed to converge:
# LexViolModel_N400 = lmer(LexSemErr ~ ExpConstr_Condition  + (1| WordInput)  + (0 + ExpConstr_Condition| WordInput), data = lexviol_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
LexViolModel_N400 = lmer(LexSemErr ~ ExpConstr_Condition  + (1| WordInput), data = lexviol_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(LexViolModel_N400)
# failed to converge:
# LexViolModel_behav_ = lmer(ThresholdCrossingIteration ~ ExpConstr_Condition   + (1| WordInput)  + (0 + ExpConstr_Condition| WordInput), data = lexviol_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
LexViolModel_behav_ = lmer(ThresholdCrossingIteration ~ ExpConstr_Condition  + (1| WordInput), data = lexviol_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(LexViolModel_behav_)

lexviol_sim_accurate %>% group_by(ExpConstr_Condition) %>% summarize(n=n())

###### Simulation 5: ANTICIPATORY SEMANTIC OVERLAP #######

PE_expected = cloze_sim %>%
  filter(Cloze_name == "high_cloze") %>%
  mutate(dummy_condition = "expected")

unexp_related_sempredoverlap = sempredoverlap_sim %>%
  filter(Relatedness_name == "SemRelated") %>%
  mutate(dummy_condition = "unexp_related")


unexp_unrelated_sempredoverlap = sempredoverlap_sim %>%
  filter(Relatedness_name == "SemUnrelated") %>%
  mutate(dummy_condition = "unexp_unrelated")

final_SPO = bind_rows(
  PE_expected,
  unexp_related_sempredoverlap,
  unexp_unrelated_sempredoverlap
)

final_SPO$dummy_condition = relevel(
  factor(final_SPO$dummy_condition),
  ref = "unexp_related"
)

final_SPO_accurate = final_SPO %>% filter(CorrectItemCrossed ==1)
# final_SPO_accurate %>% glimpse()
#failed to converge:
# expected_semoverlap_nonoverlap_model_N400 = lmer(
#   LexSemErr ~ as.factor(dummy_condition) + (1 | WordInput) + (0 +as.factor(dummy_condition)|WordInput ),
#   data = final_SPO_accurate,
#   control = lmerControl(
#     optimizer = "bobyqa",
#     optCtrl = list(maxfun = 50000)
#   )
# )
expected_semoverlap_nonoverlap_model_N400 = lmer(
  LexSemErr ~ dummy_condition + (1 | WordInput),
  data = final_SPO_accurate,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)
summary(expected_semoverlap_nonoverlap_model_N400)
# failed to converge:
# expected_semoverlap_nonoverlap_model_behav = lmer(
#   ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1 | WordInput) + (0 +as.factor(dummy_condition)|WordInput ),
#   data = final_SPO_accurate,
#   control = lmerControl(
#     optimizer = "bobyqa",
#     optCtrl = list(maxfun = 50000)
#   )
# )
expected_semoverlap_nonoverlap_model_behav = lmer(
  ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1 | WordInput),
  data = final_SPO_accurate,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)
summary(expected_semoverlap_nonoverlap_model_behav)

##### Simulations 6 and 7: COMPETITION SIMULATIONS #####
# competition_sim %>% glimpse()
competition_sim_accurate = competition_sim %>% filter(CorrectItemCrossed == 1)
# no competitor low constraint (30%) versus unrelated-to-competitor high constraint (also 30%)

semicon_sim_nocomp = competition_sim_accurate %>% 
  filter(HasCompetitor_name == "NoCompetitor") %>% 
  filter(Constraint_name == "Low") %>% 
  filter(SemRelatedtoCompetitor_name == "Unrelated") %>% 
  mutate(dummy_condition = "no_comp")
semicon_sim_unrel = competition_sim_accurate %>% 
  filter(HasCompetitor_name == "HasCompetitor") %>% 
  filter(Constraint_name == "High") %>% 
  filter(SemRelatedtoCompetitor_name == "Unrelated") %>% 
  mutate(dummy_condition = "unrelated_to_best")
semicon_sim_rel = competition_sim_accurate %>% 
  filter(HasCompetitor_name == "HasCompetitor") %>% 
  filter(Constraint_name == "High") %>% 
  filter(SemRelatedtoCompetitor_name == "Related") %>% 
  mutate(dummy_condition = "related_to_best")
semicon_sim_exp = competition_sim_accurate %>% 
  filter(HasCompetitor_name == "NoCompetitor") %>% 
  filter(Constraint_name == "High") %>% 
  filter(SemRelatedtoCompetitor_name == "Unrelated") %>% 
  mutate(dummy_condition = "best")


# Simulation 6

temp = full_join(semicon_sim_nocomp, semicon_sim_unrel)
sim_6 = full_join(temp, semicon_sim_exp)
# final_semicon %>% glimpse()
# create the three conditions: no_comp, unrelated_to_best, related_to_best
sim_6 = sim_6 %>% mutate(dummy_condition = fct_relevel(dummy_condition,"unrelated_to_best","no_comp", "best"))
sim_6 %>% glimpse()
# sim_6_N400 = lmer(LexSemErr ~ as.factor(dummy_condition) + (1| WordInput)+(0 +as.factor(dummy_condition)|WordInput ), data = sim_6, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
sim_6_N400 = lmer(LexSemErr ~ as.factor(dummy_condition) + (1| WordInput), data = sim_6, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(sim_6_N400)

# sim_6_behav = lmer(ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1| WordInput)+(0 +as.factor(dummy_condition)|WordInput), data = sim_6, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
sim_6_behav = lmer(ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1| WordInput), data = sim_6, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(sim_6_behav)

# Simulation 7: 
sim_7 = full_join(sim_6, semicon_sim_rel)
sim_7 = sim_7 %>% mutate(dummy_condition = fct_relevel(dummy_condition,"unrelated_to_best","no_comp", "best"))
sim_7_N400 = lmer(LexSemErr ~ as.factor(dummy_condition) + (1| WordInput)+(0 +as.factor(dummy_condition)|WordInput ), data = sim_7, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
sim_7_N400 = lmer(LexSemErr ~ as.factor(dummy_condition) + (1| WordInput), data = sim_7, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(sim_7_N400)
# sim_7_behav = lmer(ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1| WordInput)+(0 +as.factor(dummy_condition)|WordInput), data = sim_7, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
sim_7_behav = lmer(ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1| WordInput), data = sim_7, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(sim_7_behav)
