import numpy as np
import pandas as pd
import time
import json
import itertools

# Import specific modules from tapas
import tapas.datasets
import tapas.generators
import tapas.attacks
import tapas.threat_models
import tapas.report

from tapas.generators import Generator

# Some fancy displays when training/testing.
import tqdm

# For defining attacks
from sklearn.ensemble import RandomForestClassifier

# Import modules from AIM
from mbi import Dataset, Domain

# Generators
from modules.myctgan import CTGAN
from modules.myctgan import DPCTGAN
from modules.myctgan import PATEGAN
from modules.aim import AIM as myAIM

################################################################################################

# Load data
data = tapas.datasets.TabularDataset(data=pd.read_csv("data/census.csv", header=1, index_col=0, dtype=str)[:5000],    # str bnecause all data types are "finite representation" in this data description
                                          description=tapas.datasets.DataDescription(json.load(open("data/census.json"))),
                                          )
# Load encoded data
encoded_data = tapas.datasets.TabularDataset(data=pd.read_csv("data/encoded_census.csv")[:5000],
                                          description=tapas.datasets.DataDescription(json.load(open("data/encoded_census.json"))),
                                          )

# Get domain of census dataset (needed for AIM)
domain = json.load(open("data/census_domain.json"))
domain = Domain(list(domain.keys()), list(domain.values()))

################################################################################################

# Instantiate generators
class AIM(myAIM, Generator):
  def __init__(self, degree, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.degree = degree

  def fit(self, dataset, degree=2, num_marginals=None, max_cells=1000):    # what is max_cells?
    self.data = dataset.data
    self.description = dataset.description
    self.degree = degree    # 1-way marginals? or 2?
    self.num_marginals = num_marginals
    self.max_cells = max_cells

    self.workload = list(itertools.combinations(domain, self.degree))    # global domain variable defined outside class?
    self.workload = [cl for cl in self.workload if domain.size(cl) <= self.max_cells]
    if self.num_marginals is not None:
        self.workload = [self.workload[i] for i in self.prng.choice(len(self.workload), self.num_marginals, replace=False)]
    self.workload = [(cl, 1.0) for cl in self.workload]

    _data = Dataset(self.data, domain)
    self.model = super().fit(_data, self.workload)

  def generate(self, num_samples):
    return tapas.datasets.TabularDataset(pd.DataFrame(super().generate(num_samples)), self.description)
  
  @property
  def label(self):
    return "AIM eps={}".format(self.epsilon)
  
generators = [CTGAN(epochs=100), 
              DPCTGAN(epsilon=1, batch_size=64, epochs=100), 
              PATEGAN(epsilon=1, batch_size=64, teacher_iters=5, student_iters=5),
              AIM(epsilon=0.5, delta=0.05, degree=1)]


# Select target records
random_index = list(np.random.randint(0, 5000, 5))

from sklearn.ensemble import IsolationForest
model_isoforest = IsolationForest()
preds = model_isoforest.fit_predict(data.data.iloc[:, 3:])
outlier_index = list(np.random.choice(np.where(preds == -1)[0], 5))

targets = random_index + outlier_index


# Helper function to construct attack
def attack(dataset, target_index, generator):
    # Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
    data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(dataset,
                        auxiliary_split=0.5, num_training_records=1000, )

    # Knowledge of the generator - typically black-box access
    sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(generator, num_synthetic_records=1000, )

    # Define threat model with attacker goal: membership inference attack on a random record
    threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
                        target_record=dataset.get_records([target_index]),
                        attacker_knowledge_generator=sdg_knowledge,
                        generate_pairs=True,
                        replace_target=True,
                        iterator_tracker=tqdm.tqdm)

    # Initialize an attacker: Groundhog attack with standard parameters
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
    summary = threat_model.test(attacker, num_samples=100)
    end = time.time()
    print("time it took to test the attacker: {}".format(end-start))

    metrics = summary.get_metrics()
    metrics["dataset"] = "Census"

    # print("Results:\n", metrics.head())
    return summary, metrics


# Try with CTGAN as generator for 10 different target records
CTGAN_summaries = []    # for effective epsilon calculation
CTGAN_metrics = pd.DataFrame()

for record in targets:
  summ, metr = attack(data, record, CTGAN(epochs=30))
  CTGAN_summaries.append(summ)
  # metrics.append(CTGAN_metrics)
  CTGAN_metrics = pd.concat([CTGAN_metrics, metr], axis=0, ignore_index=True)
  print(metr.head())


# Save plots
report = tapas.report.BinaryLabelAttackReport(pd.DataFrame(CTGAN_metrics))
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.publish("figures/CTGAN_report")
# report.compare("generator", ["dataset", "attack"], "target_id", "figures/CTGAN_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(CTGAN_summaries).publish("figures/CTGAN_report")


######################################################################################################

# DP-CTGAN eps=0.5
DPCTGAN_05_summaries = []    # for effective epsilon calculation
DPCTGAN_05_metrics = pd.DataFrame()

for record in targets:
  summ, metr = attack(encoded_data, record, DPCTGAN(epsilon=0.5, batch_size=64, epochs=30))
  DPCTGAN_05_summaries.append(summ)
  # metrics.append(DPCTGAN_metrics)
  DPCTGAN_05_metrics = pd.concat([DPCTGAN_05_metrics, metr], axis=0, ignore_index=True)
  print(metr.head())

# Save plots
report = tapas.report.BinaryLabelAttackReport(pd.DataFrame(DPCTGAN_05_metrics))
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.publish("figures/DPCTGAN_05_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(DPCTGAN_05_summaries).publish("figures/DPCTGAN_05_report")


# DP-CTGAN eps=3
DPCTGAN_3_summaries = []    # for effective epsilon calculation
DPCTGAN_3_metrics = pd.DataFrame()

for record in targets:
  summ, metr = attack(encoded_data, record, DPCTGAN(epsilon=3, batch_size=64, epochs=30))
  DPCTGAN_3_summaries.append(summ)
  # metrics.append(DPCTGAN_metrics)
  DPCTGAN_3_metrics = pd.concat([DPCTGAN_3_metrics, metr], axis=0, ignore_index=True)
  print(metr.head())

# Save plots
report = tapas.report.BinaryLabelAttackReport(pd.DataFrame(DPCTGAN_3_metrics))
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.publish("figures/DPCTGAN_3_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(DPCTGAN_3_summaries).publish("figures/DPCTGAN_3_report")


# DP-CTGAN eps=5
DPCTGAN_5_summaries = []    # for effective epsilon calculation
DPCTGAN_5_metrics = pd.DataFrame()

for record in targets:
  summ, metr = attack(encoded_data, record, DPCTGAN(epsilon=5, batch_size=64, epochs=30))
  DPCTGAN_5_summaries.append(summ)
  # metrics.append(DPCTGAN_metrics)
  DPCTGAN_5_metrics = pd.concat([DPCTGAN_5_metrics, metr], axis=0, ignore_index=True)
  print(metr.head())

# Save plots
report = tapas.report.BinaryLabelAttackReport(pd.DataFrame(DPCTGAN_5_metrics))
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.publish("figures/DPCTGAN_5_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(DPCTGAN_5_summaries).publish("figures/DPCTGAN_5_report")


######################################################################################################

# PATE-GAN eps=0.5
PATEGAN_05_summaries = []    # for effective epsilon calculation
PATEGAN_05_metrics = pd.DataFrame()

for record in targets:
  summ, metr = attack(data, record, PATEGAN(epsilon=0.5, batch_size=64, teacher_iters=5, student_iters=5))
  PATEGAN_05_summaries.append(summ)
  # metrics.append(PATEGAN_metrics)
  PATEGAN_05_metrics = pd.concat([PATEGAN_05_metrics, metr], axis=0, ignore_index=True)
  print(metr.head())

# Save plots
report = tapas.report.BinaryLabelAttackReport(pd.DataFrame(PATEGAN_05_metrics))
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.publish("figures/PATEGAN_05_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(PATEGAN_05_summaries).publish("figures/PATEGAN_05_report")


# PATE-GAN eps=3
PATEGAN_3_summaries = []    # for effective epsilon calculation
PATEGAN_3_metrics = pd.DataFrame()

for record in targets:
  summ, metr = attack(data, record, PATEGAN(epsilon=3, batch_size=64, teacher_iters=5, student_iters=5))
  PATEGAN_3_summaries.append(summ)
  # metrics.append(PATEGAN_metrics)
  PATEGAN_3_metrics = pd.concat([PATEGAN_3_metrics, metr], axis=0, ignore_index=True)
  print(metr.head())

# Save plots
report = tapas.report.BinaryLabelAttackReport(pd.DataFrame(PATEGAN_3_metrics))
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.publish("figures/PATEGAN_3_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(PATEGAN_3_summaries).publish("figures/PATEGAN_3_report")


# PATE-GAN eps=5
PATEGAN_5_summaries = []    # for effective epsilon calculation
PATEGAN_5_metrics = pd.DataFrame()

for record in targets:
  summ, metr = attack(data, record, PATEGAN(epsilon=5, batch_size=64, teacher_iters=5, student_iters=5))
  PATEGAN_5_summaries.append(summ)
  # metrics.append(PATEGAN_metrics)
  PATEGAN_5_metrics = pd.concat([PATEGAN_5_metrics, metr], axis=0, ignore_index=True)
  print(metr.head())

# Save plots
report = tapas.report.BinaryLabelAttackReport(pd.DataFrame(PATEGAN_5_metrics))
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.publish("figures/PATEGAN_5_report")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(PATEGAN_5_summaries).publish("figures/PATEGAN_5_report")


######################################################################################################

AIM(epsilon=0.5, delta=0.05, degree=1)

