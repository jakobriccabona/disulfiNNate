import pyrosetta
pyrosetta.init('-mute all')
from pyrosetta.rosetta.core.scoring import ScoreType

import menten_gcn as mg
import menten_gcn.decorators as decs

import keras
from spektral.layers import *
from keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
#from sklearn.utils import shuffle
#from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score


import numpy as np
import pandas as pd

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

def UnderSample(x_label, y_label):

    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    data_reshaped = x_label.reshape(x_label.shape[0], -1)
    data_new, y_rus = rus.fit_resample(data_reshaped, y_label)
    num_features = x_label.shape[1:]
    x_rus = data_new.reshape(-1, *num_features)
    
    return x_rus, y_rus

X_rus, y_rus = UnderSample(X_train, y_train)
A_rus, y_rus = UnderSample(A_train, y_train)
E_rus, y_rus = UnderSample(E_train, y_train)

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

X_in, A_in, E_in = data_maker.generate_XAE_input_layers()

# Define GCN model
X_in, A_in, E_in = data_maker.generate_XAE_input_layers()

L1 = ECCConv(64, activation=None)([X_in, A_in, E_in])
L1_bn = BatchNormalization()(L1)
L1_act = Activation('relu')(L1_bn)
L1_drop = Dropout(0.2)(L1_act)

L2 = ECCConv(32, activation=None)([L1_drop, A_in, E_in])
L2_bn = BatchNormalization()(L2)
L2_act = Activation('relu')(L2_bn)
L2_drop = Dropout(0.2)(L2_act)

L3 = ECCConv(16, activation=None)([L2_drop, A_in, E_in])
L3_bn = BatchNormalization()(L3)
L3_act = Activation('relu')(L3_bn)
L3_drop = Dropout(0.2)(L3_act)

L3 = GlobalMaxPool()(L3_drop)
L4 = Flatten()(L3)
output = Dense(1, name="out", activation="sigmoid")(L4)

model = Model(inputs=[X_in, A_in, E_in], outputs=output)
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy')
model.summary()

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    verbose=2,
    patience=20,
    mode='min',
    restore_best_weights=True
)
history = model.fit(x=[X_rus, A_rus, E_rus], y=y_rus, batch_size=50, epochs=500, validation_data=([X_val, A_val, E_val], y_val), callbacks=[early_stopping])
model.save("v1.keras")


#validation
y_pred_prob = model.predict([X_val, A_val, E_val])
y_pred = (y_pred_prob > 0.5).astype(int) 

mcc = matthews_corrcoef(out_test, y_pred)
print('Matthews Coefficient:', mcc)
fpr, tpr, thresholds = roc_curve(out_test, y_pred_prob)
auc = roc_auc_score(y_val, y_pred_prob)
print('auc:', auc)
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
cm = confusion_matrix(y_val, y_pred)
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#Plottings
#plt.plot(history.history['loss'], label='Training Loss', color='royalblue')
#plt.plot(history.history['val_loss'], label='Validation Loss', color='goldenrod')
#plt.title('Model Loss Over Epochs')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.savefig('history.png')
#plt.clf()

#plt.plot(recall, precision)
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Precision-Recall Curve')
#plt.savefig('precision-recall.png')
#plt.clf()

#plt.plot(fpr, tpr, label='ROC curve', color='royalblue')
#plt.xscale('log')  # Set x-axis to logarithmic scale
#plt.xlabel('Log False Positive Rate (FPR)')
#plt.ylabel('True Positive Rate (TPR)')
#plt.xlim([0.00001, 1])  # Set limits for the logarithmic scale
#plt.ylim([0.0, 1.05])
#plt.title('Log ROC Curve')
#plt.savefig('log-roc.png')
#plt.clf()

#sns.heatmap(cmn, annot=True, fmt='g', cmap='Blues', xticklabels=['Non-Proline', 'Proline'], yticklabels=['Non-Proline', 'Proline'])
#plt.xlabel('Predicted Label')
#plt.ylabel('True Label')
#plt.title('Confusion Matrix')
#plt.savefig('conf-m.png')

