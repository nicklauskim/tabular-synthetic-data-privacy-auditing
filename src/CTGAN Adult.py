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
data = tapas.datasets.TabularDataset(data=pd.read_csv("data/adult.csv", header = 0, index_col=None, dtype=str)[:],    # str bnecause all data types are "finite representation" in this data description
                                          description=tapas.datasets.DataDescription(json.load(open("data/adult.json"))),
                                          )
print(data.data.head())
print(data.data.shape)

################################################################################################

# Use CTGAN as generator
from modules.myctgan import CTGAN
# Instantiate generator
generator = CTGAN(epochs=1)

from modules.myctgan import DPCTGAN
generator = DPCTGAN(epsilon=1, batch_size=64, epochs=1)    # what is batch_size?

from modules.myctgan import PATEGAN
generator = PATEGAN(epsilon=1, batch_size=64, teacher_iters=5, student_iters=5)    # what is batch_size?

# Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                    auxiliary_split=0.5, num_training_records=1000, )
# data_knowledge = tapas.threat_models.ExactDataKnowledge(data)

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

################################################################################################

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
CTGAN_metrics_0["dataset"] = "Adult"
print("Results:\n", CTGAN_metrics_0.head())

# Save plots
tapas.report.BinaryLabelAttackReport(pd.DataFrame(CTGAN_metrics_0)).publish("figures/CTGAN_0_report")

# Save report
tapas.report.EffectiveEpsilonReport([CTGAN_summary_0, CTGAN_summary_0]).publish("figures/CTGAN_0_report")

################################################################################################

# Now try attacking an outlier record rather than just a random record
from sklearn.ensemble import IsolationForest
model_isoforest = IsolationForest()
preds = model_isoforest.fit_predict(data.data.iloc[:, 3:])
#scores = model_isoforest.score_samples(data.data)
outlier_index = np.where(preds == -1)[0]

# Define new threat model aimed at an outlier target record
threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
                    target_record=data.get_records([outlier_index[1]]),    # !!!!!
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


