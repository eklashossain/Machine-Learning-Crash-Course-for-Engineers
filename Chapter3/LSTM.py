#Source: https://www.tensorflow.org/text/tutorials/text_classification_rnn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt


# --------------------Load IMDB Review Dataset-------------------
imdb_dataset = tfds.load('imdb_reviews', as_supervised=True)
train_ds, test_ds = imdb_dataset['train'], imdb_dataset['test']   # Split train and test data

# Declare buffer size to avoid overlap in data processing
BUFFER_SIZE = 100000
BATCH_SIZE = 64
# Shuffle the data and confugure for performance
autotune = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(autotune)
test_ds = test_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(autotune)

# Declare vocabulary size
Vocabulary_size = 1000
# Vectorization to convert text into corresponding number
vectorization = TextVectorization(max_tokens=Vocabulary_size)
# Extract only text from train data
train_text = train_ds.map(lambda text, labels: text)
# Map text to number using vectorization
vectorization.adapt(train_text)

# Creating callback to stop after 87% accuracy of the model
THRESHOLD = 0.87


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > THRESHOLD):
            print(f"Reached {THRESHOLD*100}% accuracy")
            self.model.stop_training = True


# --------------------Define the LSTM model----------------------
model = tf.keras.Sequential([
                            vectorization,
                            Embedding(input_dim=len(vectorization.get_vocabulary()),
                                      output_dim=32, mask_zero=True),
                            LSTM(32),
                            Dropout(0.2),
                            Dense(32, activation=tf.nn.relu),
                            Dense(1)
                            ])

# Compile defined model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])

# Fit the model on training data
history = model.fit(train_ds, epochs=10, callbacks=[myCallback()])

model.summary() # Summary of the model


# ---------Plot Loss and Accuracy with respect to Epochs---------
fontsize = 20
linewidth = 3
plt.figure(figsize=(8, 8))
plt.plot(history.history['accuracy'], color="green", linewidth=linewidth)
plt.plot(history.history['loss'], color="red", linewidth=linewidth)
plt.xlabel("Epochs", fontsize=fontsize)
plt.ylabel("Accuracy, Loss", fontsize=fontsize)
plt.legend(["Accuracy", "Loss"], fontsize=fontsize)
plt.ylim(0, 1)
plt.grid()
plt.show()

# Evaluate loss and accuracy over test dataset
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test loss:{test_loss}, Test accuracy:{test_acc}")


# -------------------Function for Prediction---------------------
def predict(text):
    predictions = model.predict(np.array([text]))
    if predictions >= 0:
        print("Positive review!!")
    else:
        print("Negative review!!")

# Predict sentiment for given review
text = """Completely time waste!! Don't waste your time, rather sleeping is better"""
predict(text)
text = """I love beautiful movies. If a film is eye-candy with carefully designed decorations, 
          masterful camerawork, lighting, and architectural frames, I can forgive anything else in…"""
predict(text)
