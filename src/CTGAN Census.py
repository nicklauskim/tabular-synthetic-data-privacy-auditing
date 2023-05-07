import numpy as np
import pandas as pd
import time
import json

# Import specific modules from tapas
import tapas.datasets
import tapas.generators
import tapas.attacks
import tapas.threat_models
import tapas.report

# Some fancy displays when training/testing.
import tqdm

# For defining attacks
from sklearn.ensemble import RandomForestClassifier

################################################################################################

# Load data
data = tapas.datasets.TabularDataset(data=pd.read_csv("data/census.csv", header=1, index_col=0, dtype=str)[:5000],    # str bnecause all data types are "finite representation" in this data description
                                          description=tapas.datasets.DataDescription(json.load(open("data/census.json"))),
                                          )

################################################################################################

# Use CTGAN(s) as generator
from modules.myctgan import CTGAN
from modules.myctgan import DPCTGAN
  
# Instantiate generator
generator = CTGAN(epochs=1)
# generator = DPCTGAN(epsilon=1, batch_size=64, epochs=100)    # what is batch_size?


################################################################################################
#Everything below this line is "held constant"
# for gen in generators:
#     gen_label = ""
#     summaries = []    # for effective epsilon calculation
#     for record in records:
#         rec_label = ""
#         ...
#         summary = threat_model.test(attacker, num_samples=100)
#         summaries.append(summary)

#         # Output nice, printable metrics that evaluate the attack
#         metrics = summary.get_metrics()
#         metrics["dataset"] = "Census"
#         print("Results:\n", metrics.head())
#         # Save plots
#         plot_path = "figures/" + gen_label + rec_label + ""
#         tapas.report.BinaryLabelAttackReport(pd.DataFrame(metrics)).publish(plot_path)

#         # Save eff eps report
#         tapas.report.EffectiveEpsilonReport(summaries).publish(plot_path)

#         # Try ROC plots?


# Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                    auxiliary_split=0.5, num_training_records=1000, )

# Knowledge of the generator - typically black-box access
sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(generator, num_synthetic_records=1000, )

# Define threat model with attacker goal: membership inference attack on a random record
threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
                    target_record=data.get_records([0]),
                    attacker_knowledge_generator=sdg_knowledge,
                    generate_pairs=True,
                    replace_target=True,
                    iterator_tracker=tqdm.tqdm)

# Initialize an attacker: Groundhog attack with standard parameters
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
feature_set = tapas.attacks.NaiveSetFeature() + tapas.attacks.HistSetFeature() + tapas.attacks.CorrSetFeature()
feature_classifier = tapas.attacks.FeatureBasedSetClassifier(feature_set, random_forest)
attacker = tapas.attacks.GroundhogAttack(feature_classifier)


# Train the attack
start = time.time()
attacker.train(threat_model, num_samples=100)
end = time.time()
print("time it took to train the attacker: {}".format(end-start))

# Test the attack
start = time.time()
CTGAN_summary_0 = threat_model.test(attacker, num_samples=100)
end = time.time()
print("time it took to test the attacker: {}".format(end-start))

# Output nice, printable metrics that evaluate the attack
CTGAN_metrics_0 = CTGAN_summary_0.get_metrics()
CTGAN_metrics_0["dataset"] = "Census"
print("Results:\n", CTGAN_metrics_0.head())













################################################################################################

# Define a new threat model for "control"
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                    auxiliary_split=0.5, num_training_records=100, )

threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
                    target_record=data.get_records([0]),
                    attacker_knowledge_generator=sdg_knowledge,
                    generate_pairs=True,
                    replace_target=True,
                    iterator_tracker=tqdm.tqdm)

# Test the attack on control group
start = time.time()
CTGAN_summary_0 = threat_model.test(attacker, num_samples=100)
end = time.time()
print("time it took to test the attacker: {}".format(end-start))

# Output metrics for control group
CTGAN_metrics_0 = CTGAN_summary_0.get_metrics()
CTGAN_metrics_0["dataset"] = "Census"
print("Results:\n", CTGAN_metrics_0.head())

# Save plots
tapas.report.BinaryLabelAttackReport(pd.DataFrame(CTGAN_metrics_0)).publish("figures/CTGAN_0_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport([CTGAN_summary_0, CTGAN_summary_0]).publish("figures/CTGAN_0_report")

################################################################################################

# Now try attacking OUTLIER records
from sklearn.ensemble import IsolationForest
model_isoforest = IsolationForest()
preds = model_isoforest.fit_predict(data.data.iloc[:, 3:])
outlier_index = np.where(preds == -1)[0]

# Define new threat model aimed at an outlier target record
threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
                    target_record=data.get_records([outlier_index[0]]),
                    attacker_knowledge_generator=sdg_knowledge,
                    generate_pairs=True,
                    replace_target=True,
                    iterator_tracker=tqdm.tqdm)

# Train the attack
start = time.time()
attacker.train(threat_model, num_samples=1000)
end = time.time()
print("time it took to train the attacker: {}".format(end-start))

# Test the attack
start = time.time()
CTGAN_summary_outlier = threat_model.test(attacker, num_samples=100)
end = time.time()
print("time it took to test the attacker: {}".format(end-start))

# Output nice, printable metrics that evaluate the attack
CTGAN_metrics_outlier = CTGAN_summary_outlier.get_metrics()
print("Results:\n", CTGAN_metrics_outlier.head())

# Save plots
tapas.report.BinaryLabelAttackReport(pd.DataFrame([CTGAN_metrics_0, CTGAN_metrics_outlier])).publish("figures/CTGAN_outlier_report")

# Save report
tapas.report.EffectiveEpsilonReport([CTGAN_summary_0, CTGAN_summary_outlier]).publish("figures/CTGAN_outlier_report")


