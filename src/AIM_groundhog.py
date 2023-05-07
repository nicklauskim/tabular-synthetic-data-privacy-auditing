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

# Import modules from AIM
from mbi import Dataset, Domain

################################################################################################

# Load data
data = tapas.datasets.TabularDataset(data=pd.read_csv("data/census.csv", header = 1, dtype=str)[:5000],    # str bnecause all data types are "finite representation" in data description
                                          description=tapas.datasets.data_description.DataDescription(json.load(open("data/census.json"))),
                                          )

# Create numerically encoded census data (needed in this format for AIM)
encoded_data = data.data.copy()
encoded_data['Residence Type'] = encoded_data['Residence Type'].astype('category').cat.codes
encoded_data['Region'] = encoded_data['Region'].astype('category').cat.codes
# encoded_data = encoded_data.astype('int32')

# Save new encoded data to a csv file
encoded_data.to_csv('data/encoded_census.csv', index=False)

# Load data
data = tapas.datasets.TabularDataset(data=pd.read_csv("data/encoded_census.csv")[:10000],
                                          description=tapas.datasets.DataDescription(json.load(open("examples/data/encoded_census.json"))),
                                          )

# Get domain of census dataset (needed for AIM)
domain = json.load(open("data/census_domain.json"))
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
  
  @property
  def label(self):
    return "AIM eps={}".format(self.epsilon)
  
# Instantiate generator
generator = AIM(epsilon=0.5, delta=0.05, degree=1)

# Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(data,
                    auxiliary_split=0.8, num_training_records=100, )

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
AIM_summary = threat_model.test(attacker, num_samples=100)
end = time.time()
print("time it took to test the attacker: {}".format(end-start))

# Output nice, printable metrics that evaluate the attack
AIM_metrics = AIM_summary.get_metrics()
print("Results:\n", AIM_metrics.head())

# Loop through for different choices of (theoretical) epsilon
list_of_metrics = []
list_of_summaries = []

epsilons = [0.5, 3, 5]
# for eps in epsilons:
#   generator = AIM(epsilon=eps, delta=0.05)
#   threat_model = tapas.threat_models.TargetedMIA(attacker_knowledge_data=data_knowledge,
#                     target_record=data.get_records([0]),
#                     attacker_knowledge_generator=sdg_knowledge,
#                     generate_pairs=True,
#                     replace_target=True,
#                     iterator_tracker=tqdm.tqdm)
#   attacker.train(threat_model, num_samples=10)
#   AIM_summary = threat_model.test(attacker, num_samples=10)
#   AIM_metrics = AIM_summary.get_metrics()
#   print()
#   print("Results:\n", AIM_metrics.head())

# # Save plots
# tapas.report.BinaryLabelAttackReport(pd.DataFrame(list_of_metrics)).publish("./figures/AIM_report")

# # Save report
# tapas.report.EffectiveEpsilonReport(list_of_summaries).publish("./figures/AIM_report")


