
from evaluationMetrics import *
import tensorflow as tf
from keras_contrib.layers import CRF
from keras.models import load_model,model_from_json
# test save/reload model.
# serialize model to JSON
def save_model(model):
  model_json = model.to_json()
  with open("/home/mrim/affim/Random-BGC-For-Arabic/model/model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("/home/mrim/affim/Random-BGC-For-Arabic/model/model.h5")
  print("Saved model to disk")
 
# later...
def test_model(x_test,z_test):
 # load json and create model
 json_file = open("/home/mrim/affim/Random-BGC-For-Arabic/model/model.json", 'r')
 loaded_model_json = json_file.read()
 json_file.close()
 loaded_model = model_from_json(loaded_model_json,custom_objects={'CRF':CRF})
 # load weights into new model
 loaded_model.load_weights("/home/mrim/affim/Random-BGC-For-Arabic/model/model.h5")
 print("Loaded model from disk")
 
 # evaluate loaded model on test data
 loaded_model.compile(tf.keras.optimizers.Adam(learning_rate=1e-5),loss="categorical_crossentropy", metrics= ['accuracy',recall_m,precision_m,f1_m])
 score = loaded_model.evaluate(x_test,z_test, verbose=0)
 #print(score)
 print((loaded_model.metrics_names[1],loaded_model.metrics_names[2],loaded_model.metrics_names[3],loaded_model.metrics_names[4], score[1]*100,score[2]*100,score[3]  *100,score[4]*100))