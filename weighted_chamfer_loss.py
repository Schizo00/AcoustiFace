import tensorflow as tf
import numpy as np
from scipy.spatial import KDTree
from tensorflow_graphics.nn.loss.hausdorff_distance import evaluate


class WeightedChamferLoss(tf.keras.losses.Loss):
  def __init__(self, attention_ids, weight, batch_size):
    super(WeightedChamferLoss, self).__init__(name='weighted_chamfer_loss')
    # self.base_loss = base_loss
    self.batch_size = batch_size
    self.weights = np.zeros((5023, 3))

    for i in attention_ids:
        self.weights[i] = (1 * 2) ** weight

    self.weights = self.weights.flatten()
    self.weights = tf.cast(self.weights, dtype=tf.float32) 

  def call(self, y_true, y_pred):
    y_true = (tf.reshape((y_true * self.weights), [self.batch_size, 5023, 3]) 
              # * 201.41335
              )
    y_pred = (tf.reshape((y_pred * self.weights), [self.batch_size, 5023, 3]) 
              # * 201.41335
              )


    

    # y_true = tf.make_tensor_proto(y_true)
    # y_pred = tf.make_tensor_proto(y_pred)

    # y_true = tf.make_ndarray(y_true)
    # y_pred = tf.make_ndarray(y_pred)


    # Calculate standard loss
    # loss = self.base_loss(y_true, y_pred)

    # Apply weights element-wise
    # weighted_loss = loss * self.weights

    return evaluate(y_true, y_pred)
    # return tf.reduce_mean(weighted_loss)
  
  def get_weights(self):
     return self.weights
  

  
  


  