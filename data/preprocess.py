import tensorflow as tf

# In the batch process, the data set is divided into parts according to the value determined as the batch value
#  and the training of the model is carried out on this part in each iteration.
def prepare_batch(x_batch, y_batch):
    
    atom_features, bond_features, pair_indices = x_batch

    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)


    
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()


    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

# Now, we pass the function defined above to the map() function in the return statement of the function below
# similar to the decorator design pattern here, so that the X, y parameters are passed to the prepare_batch() function. 
# so we can use the function to get our dataset in the future
def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)