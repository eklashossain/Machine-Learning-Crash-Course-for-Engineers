# -----------------Importing Required Libraries------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


# ------------------------Model Formation------------------------
def modified_efficientNet(image_shape=(32, 32, 3), fc_units=(1024, 512)):
    input_src = Input(shape=image_shape)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=image_shape
    )(input_src)
    x = GlobalAveragePooling2D()(base_model)

    for units in fc_units:
        x = Dense(units, activation='relu')(x)

    output = Dense(10, activation='softmax')(x)

    model = Model(input_src, output)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

    return x_train, y_train, x_test, y_test


# -----------------------Model Training--------------------------
def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=32):
    predicted_output = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
    )
    return predicted_output

def plot_accuracy(predicted_output):
    plt.plot(predicted_output.history['accuracy'])
    plt.plot(predicted_output.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.savefig("./results/train_test.png")
    plt.show()


# -----------------Detection for Unseen Images-------------------
def predict_custom_image(model, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    predicted_label_index = np.argmax(preds)
    class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    my_image = plt.imread(img_path)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(my_image)
    plt.title('Predicted: {}'.format(class_labels[predicted_label_index]))
    plt.savefig("./results/predicted.png")
    plt.show()


# -------------------Training and Evaluation---------------------
EffNet = modified_efficientNet()
print(EffNet.summary())

# Split Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

predicted_output = train_model(EffNet, x_train, y_train, x_test, y_test)

# Plot training and validation accuracy
plot_accuracy(predicted_output)

# Evaluate the model on the test dataset
test_loss, test_acc = EffNet.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


# Predict an unknown image
predict_custom_image(EffNet, './data/img.jpg')