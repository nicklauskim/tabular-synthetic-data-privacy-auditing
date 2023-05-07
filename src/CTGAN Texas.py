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
df = pd.read_csv("data/texas-orig.csv", index_col=0, dtype=str)
cols = df.columns
dtype_dict = {"DISCHARGE": "str", **{col: 'str' for col in cols[1:12]}, **{col: 'float64' for col in cols[12:]}}
df = pd.read_csv("data/texas-orig.csv", index_col=0, dtype=dtype_dict)    # change!
# print(df.dtypes)

data = tapas.datasets.TabularDataset(data=df, description=tapas.datasets.DataDescription(json.load(open("data/texas.json"))))

################################################################################################

from modules.myctgan import CTGAN
from modules.myctgan import DPCTGAN

# Instantiate generator
generator = CTGAN(epochs=1)
generator = DPCTGAN(epsilon=1, batch_size=64, epochs=1)

# Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                    auxiliary_split=0.5, num_training_records=100, )

# Knowledge of the generator - typically black-box access
sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(generator, num_synthetic_records=100, )

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
print("Results:\n", CTGAN_metrics_0.head())

# Save plots
CTGAN_report_0 = tapas.report.BinaryLabelAttackReport(pd.DataFrame(CTGAN_metrics_0))
CTGAN_report_0.publish("figures/CTGAN_0_texas_report")

CTGAN_report_0.metrics = ["privacy_gain", "auc", "effective_epsilon"]
CTGAN_report_0.compare("generator", ["dataset", "attack"], "target_id", "figures/CTGAN_0_texas_report")

# Save report
tapas.report.EffectiveEpsilonReport([CTGAN_summary_0, CTGAN_summary_0]).publish("figures/CTGAN_0_texas_report")

################################################################################################

# Now try attacking an outlier record rather than just a random record from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest
model_isoforest = IsolationForest()
preds = model_isoforest.fit_predict(data.data.iloc[:, 7:])
#scores = model_isoforest.score_samples(data.data)
outlier_index = np.where(preds == -1)[0]

CTGAN_metrics = pd.DataFrame()
CTGAN_summaries = []

for i in range(5):
  # Instantiate generator
  generator = CTGAN(epochs=1)

  # Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
  data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                      auxiliary_split=0.5, num_training_records=100, )

  # Knowledge of the generator - typically black-box access
  sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(generator, num_synthetic_records=100, )

  # Initiate threat model
  threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
                      target_record=data.get_records([outlier_index[i]]),
                      attacker_knowledge_generator=sdg_knowledge,
                      generate_pairs=True,
                      replace_target=True,
                      iterator_tracker=tqdm.tqdm)

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
  CTGAN_summary_outlier = threat_model.test(attacker, num_samples=100)
  end = time.time()
  print("time it took to test the attacker: {}".format(end-start))

  # Output nice, printable metrics that evaluate the attack
  CTGAN_metrics_outlier = CTGAN_summary_outlier.get_metrics()
  print("Results:\n", CTGAN_metrics_outlier.head())

  # Save summary
  CTGAN_metrics_outlier["dataset"] = "Texas"
  CTGAN_metrics = pd.concat([CTGAN_metrics, CTGAN_metrics_outlier], axis=0, ignore_index=True)
  CTGAN_summaries.append(CTGAN_summary_outlier)

# Save plot with all outlier summaries
tapas.report.BinaryLabelAttackReport(CTGAN_metrics).publish("figures/CTGAN_outlier_texas_report")

# Save report
tapas.report.EffectiveEpsilonReport(CTGAN_summaries).publish("figures/CTGAN_outlier_texas_report")


