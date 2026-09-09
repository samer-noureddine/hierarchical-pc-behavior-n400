library(lme4)
library(lmerTest)
library(tidyverse)
library(emmeans)
library(effectsize)


# prevent R from wrapping its output around too soon
options(width=300)

nrm <- function(x) {
  return((x - mean(x))/sd(x))
}

base = "C:/Users/samer/postdoc_projects/~hierarchicalPC/comprehension/PC_KLdiv_multiplicative_PCBCDIM/simulation_csv_files//"

# load data
std_sim= read_csv(paste(base, "Standard_Simulation_N400_2_to_6_IterationsToThreshold0.6.csv", sep = ""))
# wrd_psd_sim= read_csv(paste(base, "Word_vs_Pseudoword_Simulation_N400_2_to_6_IterationsToThreshold0.6.csv", sep = ""))
reppriming_sim= read_csv(paste(base, "RepetitionPriming_Simulation_N400_27_to_31_IterationsToThreshold0.6.csv", sep = ""))
sempriming_sim = read_csv(paste(base, "SemanticPriming_Simulation_N400_27_to_31_IterationsToThreshold0.6.csv", sep = ""))
# formpriming_sim = read_csv(paste(base, "FormPriming_Simulation_N400_27_to_36_IterationsToThreshold0.6.csv", sep = ""))
# wrdpsd_formpriming_sim = read_csv(paste(base, "FormPriming_WrdPsd_Simulation_N400_27_to_36_IterationsToThreshold0.6.csv", sep = ""))
cloze_sim = read_csv(paste(base, "ClozeProbability_Simulation_N400_22_to_26_IterationsToThreshold0.6.csv", sep = ""))
lexviol_sim = read_csv(paste(base, "LexicalViolation_Simulation_N400_22_to_26_IterationsToThreshold0.6.csv", sep = ""))
sempredoverlap_sim= read_csv(paste(base, "SemanticPredictionOverlap_Simulation_N400_22_to_26_IterationsToThreshold0.6.csv", sep = ""))
competition_sim= read_csv(paste(base, "Competition_Simulation_N400_22_to_26_IterationsToThreshold0.6.csv", sep = ""))
# orthpredoverlap_sim= read_csv(paste(base, "OrthographicPredictionOverlap_Simulation_N400_22_to_31_IterationsToThreshold0.6.csv", sep = ""))


# normalize all quantitative variables
nrm_std_sim = std_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_reppriming_sim = reppriming_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_sempriming_sim = sempriming_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_cloze_sim = cloze_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_lexviol_sim = lexviol_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_sempredoverlap_sim = sempredoverlap_sim %>% mutate_if(is.numeric, list(norm = nrm))
nrm_competition_sim = competition_sim %>% mutate_if(is.numeric, list(norm = nrm))
options(width = 300)

######## stats ########
numstims = nrm_std_sim %>% summarize(n = n()) %>% pull(n)
# Section 1.1, 1.3, 1.4
nrm_std_sim_accurate = nrm_std_sim %>% filter(CorrectItemCrossed == 1)
nrm_std_sim_accurate %>% summarize(n = n(), meanlse = mean(LexSemErr), meantci = mean(ThresholdCrossingIteration))
std_model_N400 = lm(LexSemErr ~ ONsize_norm + Frequency_norm + NumSemFeats_norm, data = nrm_std_sim_accurate)
nrm_std_sim %>% summarize(n = n(), meanlse = mean(LexSemErr), meantci = mean(ThresholdCrossingIteration))
nrm_std_sim %>% filter(ThresholdWasCrossed == TRUE) %>% group_by(CorrectItemCrossed) %>% summarize(n = n(), meanlse = mean(LexSemErr), meantci = mean(ThresholdCrossingIteration))
options(width = 100)
# options(repr.plot.width=30, repr.plot.height=8)
ggplot(nrm_std_sim_accurate, aes(x = Frequency_norm, y = LexSemErr)) +
                geom_point() +  # Add points
                labs(x = "Frequency", y = "LexSemErr") +  # Set axis labels
                theme_minimal()



summary(std_model_N400)
std_model_behav = lm(ThresholdCrossingIteration ~ ONsize_norm + Frequency_norm + NumSemFeats_norm, data = nrm_std_sim_accurate)
summary(std_model_behav)

# Section 2.1
options(width = 100)
nrm_reppriming_sim %>% glimpse()
nrm_reppriming_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(Repeated_code) %>% summarize(accuracy = sum(CorrectItemCrossed)/n())

nrm_reppriming_sim_accurate = nrm_reppriming_sim %>% filter(CorrectItemCrossed == 1)
nrm_reppriming_sim_accurate %>% group_by(Repeated_name) %>% summarize(n=n())
nrm_reppriming_sim %>% group_by(Repeated_name) %>% summarize(n=n())
# failed to converge:
# RepPrimModel = lmer(LexSemErr ~ Repeated_code + (Repeated_code | WordInput), data = nrm_reppriming_sim, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
RepPrimModel_N400 = lmer(LexSemErr ~ Repeated_code + (1 | WordInput), data = nrm_reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(RepPrimModel_N400)
# failed to converge:
# RepPrimModel_behav = lmer(ThresholdCrossingIteration ~ Repeated_code + (Repeated_code || WordInput), data = nrm_reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
RepPrimModel_behav = lmer(ThresholdCrossingIteration ~ Repeated_code + (1 | WordInput), data = nrm_reppriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(RepPrimModel_behav)


options(width = 100)
nrm_sempriming_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(SemanticRelatedness_code) %>% summarize(accuracy = sum(CorrectItemCrossed)/n())

# Section 2.2
nrm_sempriming_sim_accurate = nrm_sempriming_sim %>% filter(CorrectItemCrossed == 1)
nrm_sempriming_sim_accurate %>% group_by(SemanticRelatedness_name) %>% summarize(n=n()/numstims)
# failed to converge:
# SemPrimModel_N400= lmer(LexSemErr ~ SemanticRelatedness_code + (SemanticRelatedness_code || WordInput), data = nrm_sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
SemPrimModel_N400= lmer(LexSemErr ~ SemanticRelatedness_code + (1 | WordInput), data = nrm_sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SemPrimModel_N400)
# SemPrimModel_behav= lmer(ThresholdCrossingIteration ~ SemanticRelatedness_code + (SemanticRelatedness_code || WordInput), data = nrm_sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
SemPrimModel_behav= lmer(ThresholdCrossingIteration ~ SemanticRelatedness_code + (1 | WordInput), data = nrm_sempriming_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(SemPrimModel_behav)



# difference between repeated and semantically primed words
repprimed = nrm_reppriming_sim_accurate %>% filter(Repeated_name == "repeated")%>% mutate(dummy_condition = "repeated")
semprimed = nrm_sempriming_sim_accurate %>% filter(SemanticRelatedness_name == "semrelated") %>% mutate(dummy_condition = "semrelated")
sem_vs_rep = full_join(semprimed, repprimed, by = "WordInds")
t.test(sem_vs_rep$LexSemErr.x, sem_vs_rep$LexSemErr.y, paired = TRUE)
t.test(sem_vs_rep$ThresholdCrossingIteration.x, sem_vs_rep$ThresholdCrossingIteration.y, paired = TRUE)

sem_vs_rep %>% glimpse()


# Section 3.1
nrm_cloze_sim %>% filter(ThresholdWasCrossed == 1) %>% group_by(as.factor(Cloze_code)) %>% summarize(accuracy = sum(CorrectItemCrossed)/n(), n = n())
nrm_cloze_sim_accurate = nrm_cloze_sim %>% filter(CorrectItemCrossed == 1)
nrm_cloze_sim_accurate %>% group_by(Cloze_code) %>% summarize(n=n()/numstims)
nrm_cloze_sim %>% group_by(as.factor(Cloze_code)) %>% summarize(accuracy = sum(ThresholdWasCrossed)/n(), n = n())

nrm_cloze_sim%>% glimpse()
ClzModel_N400 = lmer(LexSemErr ~ Cloze_code_norm + (Cloze_code_norm  | WordInput), data = nrm_cloze_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(ClzModel_N400)
ClzModel_behav = lmer(ThresholdCrossingIteration ~ Cloze_code_norm + (Cloze_code_norm  | WordInput), data = nrm_cloze_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(ClzModel_behav)
# Section 3.2

nrm_lexviol_sim_accurate = nrm_lexviol_sim %>% filter(CorrectItemCrossed == 1)
nrm_lexviol_sim_accurate %>% glimpse()
nrm_lexviol_sim_accurate$ExpConstr_Condition <- factor(
  paste(nrm_lexviol_sim_accurate$Constraint_name, nrm_lexviol_sim_accurate$IsExpected_name, sep = "_")
)

nrm_lexviol_sim_accurate$ExpConstr_Condition <- relevel(
  nrm_lexviol_sim_accurate$ExpConstr_Condition,
  ref = "HighConstraint_Expected"
)
# failed to converge:
# LexViolModel_N400_ = lmer(LexSemErr ~ ExpConstr_Condition  + (ExpConstr_Condition|| WordInput), data = nrm_lexviol_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
LexViolModel_N400_ = lmer(LexSemErr ~ ExpConstr_Condition  + (1| WordInput), data = nrm_lexviol_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(LexViolModel_N400_)
LexViolModel_behav_ = lmer(ThresholdCrossingIteration ~ ExpConstr_Condition  + (1| WordInput), data = nrm_lexviol_sim_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(LexViolModel_behav_)


nrm_lexviol_sim_unexp_accurate = nrm_lexviol_sim %>% filter(IsExpected_code == -0.5) %>% filter(CorrectItemCrossed == 1)
nrm_lexviol_sim_unexp_accurate %>% group_by(WordInput, Constraint_code) %>% summarize(n=n()/numstims, tci = mean(ThresholdCrossingIteration))
nrm_lexviol_sim_unexp_accurate %>% glimpse()

# bad_list <- scan("C:/Users/samer/postdoc_projects/~hierarchicalPC/comprehension/PC_KLdiv_multiplicative_PCBCDIM/bad_list.txt", what = "", quiet = TRUE)
# bad_list

# df2 <- nrm_lexviol_sim_unexp_accurate %>% filter(!WordInput %in% bad_list)
# failed to converge:
# LexViolModel_N400 = lmer(LexSemErr ~ Constraint_code  + (Constraint_code | WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
# LexViolModel_N400 = lmer(LexSemErr ~ Constraint_code  + (Constraint_code || WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
LexViolModel_N400 = lmer(LexSemErr ~ Constraint_code  + (1 | WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
nrm_lexviol_sim_unexp_accurate %>% glimpse()

summary(LexViolModel_N400)
LexViolModel_behav = lmer(ThresholdCrossingIteration ~ Constraint_code  + (Constraint_code | WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
LexViolModel_behav = lmer(ThresholdCrossingIteration ~ Constraint_code  + (Constraint_code || WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
LexViolModel_behav = lmer(ThresholdCrossingIteration ~ Constraint_code  + (1 | WordInput), data = nrm_lexviol_sim_unexp_accurate, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000)))
summary(LexViolModel_behav)
nrm_lexviol_sim_unexp_accurate %>% group_by(ThresholdWasCrossed) %>% summarize(n=n())


###### ANTICIPATORY SEMANTIC OVERLAP #######

PE_expected = nrm_cloze_sim %>%
  filter(Cloze_name == "high_cloze") %>%
  mutate(dummy_condition = "expected")

unexp_related_sempredoverlap = nrm_sempredoverlap_sim %>%
  filter(Relatedness_name == "SemRelated") %>%
  mutate(dummy_condition = "unexp_related")

nrm_sempredoverlap_sim %>% glimpse()


unexp_unrelated_sempredoverlap = nrm_sempredoverlap_sim %>%
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
final_SPO_accurate %>% glimpse()

expected_semoverlap_nonoverlap_model_N400 = lmer(
  LexSemErr ~ dummy_condition + (1 | WordInput),
  data = final_SPO_accurate,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)
summary(expected_semoverlap_nonoverlap_model_N400)

expected_semoverlap_nonoverlap_model_behav = lmer(
  ThresholdCrossingIteration ~ dummy_condition + (1 | WordInput),
  data = final_SPO_accurate,
  control = lmerControl(
    optimizer = "bobyqa",
    optCtrl = list(maxfun = 50000)
  )
)
summary(expected_semoverlap_nonoverlap_model_behav)

##### COMPETITION SIMULATIONS #####
nrm_competition_sim %>% glimpse()
nrm_competition_sim_accurate = nrm_competition_sim %>% filter(CorrectItemCrossed == 1)
# no competitor low constraint (30%) versus unrelated-to-competitor high constraint (also 30%)


semicon_sim_nocomp = nrm_competition_sim_accurate %>% 
  filter(HasCompetitor_name == "NoCompetitor") %>% 
  filter(Constraint_name == "Low") %>% 
  filter(SemRelatedtoCompetitor_name == "Unrelated") %>% 
  mutate(dummy_condition = "no_comp")
semicon_sim_unrel = nrm_competition_sim_accurate %>% 
  filter(HasCompetitor_name == "HasCompetitor") %>% 
  filter(Constraint_name == "High") %>% 
  filter(SemRelatedtoCompetitor_name == "Unrelated") %>% 
  mutate(dummy_condition = "unrelated_to_best")
semicon_sim_rel = nrm_competition_sim_accurate %>% 
  filter(HasCompetitor_name == "HasCompetitor") %>% 
  filter(Constraint_name == "High") %>% 
  filter(SemRelatedtoCompetitor_name == "Related") %>% 
  mutate(dummy_condition = "related_to_best")
semicon_sim_exp = nrm_competition_sim_accurate %>% 
  filter(HasCompetitor_name == "NoCompetitor") %>% 
  filter(Constraint_name == "High") %>% 
  filter(SemRelatedtoCompetitor_name == "Unrelated") %>% 
  mutate(dummy_condition = "best")

temp = full_join(semicon_sim_nocomp, semicon_sim_unrel)
temp = full_join(temp, semicon_sim_exp)
final_semicon = full_join(temp, semicon_sim_rel)
final_semicon %>% glimpse()
# create the three conditions: no_comp, unrelated_to_best, related_to_best
final_semicon = final_semicon %>% mutate(dummy_condition = fct_relevel(dummy_condition,"unrelated_to_best","no_comp", "related_to_best"))
semicon_model_N400 = lmer(LexSemErr ~ as.factor(dummy_condition) + (1| WordInput), data = final_semicon, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(semicon_model_N400)

semicon_model_behav = lmer(ThresholdCrossingIteration ~ as.factor(dummy_condition) + (1| WordInput), data = final_semicon, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=50000))) # nolint: line_length_linter.
summary(semicon_model_behav)



