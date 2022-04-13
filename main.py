from classes.network import MPNNModel
from data.preprocess import MPNNDataset
from classes.transition import graphs_from_smiles, molecule_from_smiles
from custommetrics.metrics import f1

from tensorflow import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import MolsToGridImage

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

np.random.seed(42)


df = pd.read_csv('data/covid-19.csv')

# dataframe karıştırılıyor
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# 80% train
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
x_train = graphs_from_smiles(df.iloc[train_index].isosmiles)
y_train = df.iloc[train_index].target

# 10% validation
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.9)]
x_valid = graphs_from_smiles(df.iloc[valid_index].isosmiles)
y_valid = df.iloc[valid_index].target

# 10% test
test_index = permuted_indices[int(df.shape[0] * 0.9) :]
x_test = graphs_from_smiles(df.iloc[test_index].isosmiles)
y_test = df.iloc[test_index].target


train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)

mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
)

mpnn.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.AUC(name="AUC"), f1]
)


history = mpnn.fit(train_dataset, validation_data=valid_dataset, epochs=20, verbose=1)

y_pred_keras = mpnn.predict(test_dataset)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('visualizations/roc curve.png')

plt.figure(figsize=(10, 6))
plt.plot(history.history["AUC"], label="train AUC")
plt.plot(history.history["val_AUC"], label="valid AUC")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("AUC", fontsize=16)
plt.legend(fontsize=16)
plt.savefig('visualizations/auc.png')

plt.figure(figsize=(10, 6))
plt.plot(history.history["f1"], label="Train F1 Score")
plt.plot(history.history["val_f1"], label="Valid F1 Score")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("F1 Score", fontsize=16)
plt.legend(fontsize=16)
plt.savefig('visualizations/f1 score.png')

molecules = [molecule_from_smiles(df.isosmiles.values[index])for index in test_index]
names = [df.cmpdname.values[index] for index in test_index]
y_true = [df.target.values[index] for index in test_index]
y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)

legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f} \n {names[i].split(',')[0]}" for i in range(0,8)]
img = MolsToGridImage(molecules[0:8], molsPerRow=4, legends=legends, returnPNG=False, subImgSize=(400, 400))
img.save("visualizations/drug results.png")
