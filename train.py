import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

import keras
import tensorflow
from spektral.layers import *
from keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

data = np.load('merged_file.npz')
Xs = data['Xs']
As = data['As']
Es = data['Es']
outs = data['outs']

Xs = np.asarray( Xs )
As = np.asarray( As )
Es = np.asarray( Es )
outs = np.asarray( outs )

# Train Test split
X_train, X_val, A_train, A_val, E_train, E_val, y_train, y_val = train_test_split(Xs, As, Es, outs, test_size=0.2, random_state=42)

# Weighted Random Over Sampling to balance classes according to natural prevalence (3% positive)
desired_pos_ratio = 0.03
n_samples = X_train.shape[0]
n_pos = int(n_samples * desired_pos_ratio)
n_neg = n_samples - n_pos

# Count current class distribution
counter = Counter(y_train.flatten())
current_pos = counter[1]
current_neg = counter[0]

# Case 1: Need to oversample positives
if current_pos < n_pos:
    sampling_strategy = {0: current_neg, 1: n_pos}
    sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
    print('\n\n Oversampling applied \n\n')

# Case 2: Need to undersample negatives
elif current_pos > n_pos:
    sampling_strategy = {0: n_neg, 1: current_pos}
    sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    print('\n\n Undersampling applied \n\n')

# Case 3: Already exactly at ratio
else:
    sampler = None  # No resampling needed
    print('\n\n No resampling needed \n\n')

def apply_sampler(sampler, X, y):
    if sampler is None:
        return X, y
    Xs_reshaped = X.reshape(X.shape[0], -1)
    X_res, y_res = sampler.fit_resample(Xs_reshaped, y)
    return X_res.reshape(-1, *X.shape[1:]), y_res

X_ros, y_ros = apply_sampler(sampler, X_train, y_train)
A_ros, _     = apply_sampler(sampler, A_train, y_train)
E_ros, _     = apply_sampler(sampler, E_train, y_train)

decorators = [decs.CACA_dist(use_nm=True),
              decs.trRosettaEdges(sincos=True, use_nm=True),
              decs.Rosetta_Ref2015_TwoBodyEneriges(individual=True, score_types=[ScoreType.fa_rep,
                                                                                 ScoreType.fa_atr, 
                                                                                 ScoreType.fa_sol, 
                                                                                 ScoreType.lk_ball_wtd, 
                                                                                 ScoreType.fa_elec, 
                                                                                 ScoreType.hbond_sr_bb, 
                                                                                 ScoreType.hbond_lr_bb, 
                                                                                 ScoreType.hbond_bb_sc, 
                                                                                 ScoreType.hbond_sc])]
data_maker = mg.DataMaker(decorators=decorators,
                           edge_distance_cutoff_A=8.0,
                           max_residues=20,
                           nbr_distance_cutoff_A=12.0)

# Define GCN model
X_in, A_in, E_in = data_maker.generate_XAE_input_layers()

X_noisy = GaussianNoise(0.1)(X_in)
E_noisy = GaussianNoise(0.1)(E_in)

L1 = ECCConv(64, activation=None)([X_noisy, A_in, E_noisy])
L1_bn = BatchNormalization()(L1)
L1_act = Activation('relu')(L1_bn)
L1_drop = Dropout(0.2)(L1_act)

L2 = ECCConv(32, activation=None)([L1_drop, A_in, E_noisy])
L2_bn = BatchNormalization()(L2)
L2_act = Activation('relu')(L2_bn)
L2_drop = Dropout(0.2)(L2_act)

L3 = ECCConv(16, activation=None)([L2_drop, A_in, E_noisy])
L3_bn = BatchNormalization()(L3)
L3_act = Activation('relu')(L3_bn)
L3_drop = Dropout(0.2)(L3_act)

L4 = GATConv(8, attn_heads=2, concat_heads=True, activation=None)([L3_drop, A_in])
L4_bn = BatchNormalization()(L4)
L4_act = Activation('relu')(L4_bn)
L4_drop = Dropout(0.2)(L4_act)

L5 = GlobalMaxPool()(L4_drop)
L6 = Flatten()(L5)
output = Dense(1, name="out", activation="sigmoid", kernel_regularizer=l2(0.01))(L6)

model = Model(inputs=[X_in, A_in, E_in], outputs=output)
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy')

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=2,
    patience=20,
    mode='min',
    restore_best_weights=True
)
history = model.fit(x=[X_ros, A_ros, E_ros], y=y_ros, batch_size=32, epochs=100, validation_data=([X_val, A_val, E_val], y_val), callbacks=[early_stopping])
model.save("v4-weighted_sampling.keras")

#validation
y_pred_prob = model.predict([X_val, A_val, E_val])
y_pred = (y_pred_prob > 0.5).astype(int)

mcc = matthews_corrcoef(y_val, y_pred)
print('Matthews Coefficient:', mcc)
fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
auc = roc_auc_score(y_val, y_pred_prob)
print('auc:', auc)
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
cm = confusion_matrix(y_val, y_pred)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#Plottings
plt.plot(history.history['loss'], label='Training Loss', color='royalblue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='goldenrod')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('history.png')
plt.clf()

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision-recall.png')
plt.clf()

plt.plot(fpr, tpr, label='ROC curve', color='royalblue')
plt.xscale('log')  # Set x-axis to logarithmic scale
plt.xlabel('Log False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.xlim([0.00001, 1])  # Set limits for the logarithmic scale
plt.ylim([0.0, 1.05])
plt.title('Log ROC Curve')
plt.savefig('log-roc.png')
plt.clf()

sns.heatmap(cmn, annot=True, fmt='g', cmap='Blues', xticklabels=['Non-Disulfide', 'Disulfide'], yticklabels=['Non-Disulfide', 'Disulfide'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('conf-m.png')