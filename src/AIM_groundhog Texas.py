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

# For defining some attacks
from sklearn.ensemble import RandomForestClassifier

# AIM
from mbi import Dataset, Domain

################################################################################################

# Load data
df = pd.read_csv("data/texas-orig.csv", index_col=0)
print(df.dtypes)

# Create numerically encoded census data (needed in this format for AIM)
for col in df.select_dtypes('object'):
  df[col] = df[col].astype('category').cat.codes
print(df.dtypes)

for col in df.select_dtypes(include=['int8', 'int64']):
  print(col)
  print(df[col].unique())

# Save new encoded data to a csv file
df.to_csv('data/encoded_texas.csv', index=False)

data = tapas.datasets.TabularDataset(data=df, description=tapas.datasets.DataDescription(json.load(open("data/encoded_texas.json"))),)

data = tapas.datasets.TabularDataset(data=pd.read_csv("data/encoded_texas.csv"),
                                        description=tapas.datasets.data_description.DataDescription(json.load(open("data/encoded_texas.json"))),)

# Get domain of dataset (needed for AIM)
domain = json.load(open("data/encoded_texas.json"))
domain = Domain(list(domain.keys()), list(domain.values()))
print(domain)

################################################################################################

from modules.aim import AIM as myAIM

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

  # The call method should map a dataset to a synthetic dataset.
  # def __call__(self, dataset, num_samples, **kwargs):
  #     self.fit(dataset, **kwargs)
  #     return self.generate(num_samples)
  
  @property
  def label(self):
    return "AIM eps={}".format(self.epsilon)
    
# Instantiate generator
generator = AIM(epsilon=0.5, delta=0.05, degree=1)

# Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                    auxiliary_split=0.5, num_training_records=100, )

# Knowledge of the generator - typically black-box access
sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(generator, num_synthetic_records=100, )

# Attacker goal: membership inference attack on a random record - try an outlier record!!
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

print("threat model defined")

################################################################################################

# Train the attack
start = time.time()
attacker.train(threat_model, num_samples=2)
end = time.time()
print("time it took to train the attacker: {}".format(end-start))

# Test the attack
start = time.time()
CTGAN_summary_0 = threat_model.test(attacker, num_samples=2)
end = time.time()
print("time it took to test the attacker: {}".format(end-start))

# Output nice, printable metrics that evaluate the attack
CTGAN_metrics_0 = CTGAN_summary_0.get_metrics()
print("Results:\n", CTGAN_metrics_0.head())

# Save plots
tapas.report.BinaryLabelAttackReport(pd.DataFrame(CTGAN_metrics_0)).publish("figures/CTGAN_0_texas_report")

# Save report
tapas.report.EffectiveEpsilonReport([CTGAN_summary_0, CTGAN_summary_0]).publish("figures/CTGAN_0_texas_report")

################################################################################################

# Now try attacking an outlier record rather than just a random record from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest
model_isoforest = IsolationForest()
preds = model_isoforest.fit_predict(data.data.iloc[:, 7:])
#scores = model_isoforest.score_samples(data.data)
outlier_index = np.where(preds == -1)[0]
print(outlier_index)

CTGAN_metrics = pd.DataFrame()
CTGAN_summaries = []

for i in range(5):
  # Instantiate generator
  generator = AIM(epochs=1)

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
  #CTGAN_summary = tapas.report.MIAttackSummary(dataset_info="Census")
  CTGAN_metrics_outlier = CTGAN_summary_outlier.get_metrics()
  print("Results:\n", CTGAN_metrics_outlier.head())


  # Save summary
  CTGAN_metrics_outlier["dataset"] = "Texas"
  CTGAN_metrics = pd.concat([CTGAN_metrics, CTGAN_metrics_outlier], axis=0, ignore_index=True)

  CTGAN_summaries.append(CTGAN_summary_outlier)


# Save plot with all outlier summaries
tapas.report.BinaryLabelAttackReport(CTGAN_metrics).publish("./figures/CTGAN_outlier_texas_report")

# Save report
tapas.report.EffectiveEpsilonReport(CTGAN_summaries).publish("./figures/CTGAN_outlier_texas_report")


