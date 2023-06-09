################################################################################################
# Import required packages + modules
################################################################################################

import numpy as np
import pandas as pd
import time
import json
import itertools
from sklearn.ensemble import IsolationForest

# Import specific modules from tapas
import tapas.datasets
import tapas.generators
import tapas.attacks
import tapas.threat_models
import tapas.report

from tapas.generators import Generator, ReprosynGenerator, Raw

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


################################################################################################
# Load data
################################################################################################

df = pd.read_csv("data/texas-big.csv", index_col=None)[:1000]
df = pd.read_csv("data/texas-orig.csv", index_col=0)
cols = df.columns
dtype_dict = {"DISCHARGE": "str", **{col: 'str' for col in cols[1:12]}, **{col: 'float64' for col in cols[12:]}}
df = df.astype(dtype_dict)

data = tapas.datasets.TabularDataset(data=df, description=tapas.datasets.DataDescription(json.load(open("data/texas-big.json"))))

# Load encoded data
encoded_data = tapas.datasets.TabularDataset(data=pd.read_csv("data/encoded_texas.csv"),
                                        description=tapas.datasets.data_description.DataDescription(json.load(open("data/encoded_texas.json"))),)

# Get domain of dataset (needed for AIM)
domain = json.load(open("data/texas_domain.json"))
domain = Domain(list(domain.keys()), list(domain.values()))
print(domain)


################################################################################################
# Attack/audit
################################################################################################

# Select random target record indices
random_index = list(np.random.randint(0, 1000, 50))

# Select outlier target record indices
model_isoforest = IsolationForest()
preds = model_isoforest.fit_predict(data.data.iloc[:, 7:])
outlier_index = list(np.random.choice(np.where(preds == -1)[0], 50))

targets = random_index + outlier_index


# Helper function to construct attack
def attack(dataset, target_index, generator):
    # Knowledge of the real data - assume auxiliary knowledge of the private dataset - auxiliary data from same distribution
    data_knowledge = tapas.threat_models.AuxiliaryDataKnowledge(dataset,
                        auxiliary_split=0.5, num_training_records=100, )

    # Knowledge of the generator - typically black-box access
    sdg_knowledge = tapas.threat_models.BlackBoxKnowledge(generator, num_synthetic_records=100, )

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
    metrics["dataset"] = "Texas"

    # print("Results:\n", metrics.head())
    return summary, metrics


# Can try attack with several generators (not all generators work on all datasets)
generators = [Raw(), 
              CTGAN(epochs=30), 
              DPCTGAN(epsilon=0.5, batch_size=64, epochs=30), 
              DPCTGAN(epsilon=1, batch_size=64, epochs=30),
              DPCTGAN(epsilon=3, batch_size=64, epochs=30)]

all_metrics = pd.DataFrame()
all_summaries = []    # for effective epsilon calculation

# Loop through all generators
for gen in generators: 
    # Loop through all target records
    for target in targets:
        try:  
            summ, metr = attack(dataset=data, target_record=target, generator=gen)
            all_summaries.append(summ)
            all_metrics = pd.concat([all_metrics, metr], axis=0, ignore_index=True)
            # print(metr.head())
        except Exception:
            continue

# Now, check metrics and summaries
print(all_metrics)
print(all_summaries)


################################################################################################
# Create plots
################################################################################################

# Get indices for only the attacks on random targets
num_attacks = all_metrics.shape[0]
random_indices = [num for i in range(0, num_attacks, 10) for num in range(i, i+5)]

# Compare generators on random targets
report = tapas.report.BinaryLabelAttackReport(all_metrics.iloc[random_indices])
report.metrics = ["privacy_gain", "auc", "effective_epsilon"]
report.compare(comparison_column='generator', fixed_pair_columns=['attack', 'dataset'], marker_column='target_id', filepath="figures/compare_generators")

# Save eff eps report
tapas.report.EffectiveEpsilonReport(selected_summaries).publish("figures/CTGAN_Texas_report")


# Compare attack for different target record types: random and outlier
all_metrics['target_type'] = (['Random']*5 + ['Outlier']*5) * len(generators)

report = tapas.report.BinaryLabelAttackReport(all_metrics)
report.metrics = ["privacy_gain", "auc"]
report.compare(comparison_column='generator', fixed_pair_columns=['attack', 'dataset'], marker_column='target_type', filepath="figures/random_versus_outlier")


