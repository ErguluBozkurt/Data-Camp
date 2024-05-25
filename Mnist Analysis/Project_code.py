import os
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import load_model
from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout



# Visualization
def plot_data(history, metric):
    train_metric = history.history[metric]
    validation_metric = history.history["val_" + metric]

    plt.plot(range(1, len(train_metric) + 1), train_metric, label="Training " + metric)
    plt.plot(range(1, len(validation_metric) + 1), validation_metric, label="Validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.show()


# Dataset Load
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(test_images.shape)


# Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_shape=(28, 28, 1))) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), activation="relu")) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(0.3)) 
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.3)) 
model.add(Dense(units=10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 

print(model.summary()) 

hist = model.fit(x=train_images, y=train_labels, batch_size=64, validation_split=0.20, epochs=50, verbose=2)

print("Best Score:", round(np.max(hist.history["accuracy"]), 2))

# Visualization
plot_data(hist, "accuracy")



# Model Save
model.save("model/numbers.h5")



# Load the trained model
model_path = "model/numbers.h5"
prediction_model = load_model(model_path)


files = os.listdir()
print(files)
os.chdir('Images')
images = os.listdir()
print(images)


for img in images:
    image = cv2.imread(img)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(gray_image, (28, 28))  # Resize to match training size
    img = img.reshape(28, 28, 1)
    predictions = prediction_model.predict(np.expand_dims(img, axis=0))
    classIndex = np.argmax(predictions)
    propVal = np.amax(predictions) 
    if(propVal>0.9):
        cv2.putText(image, f"Number:{classIndex}", (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,255,0), 1)
    
    cv2.imshow("Image", image)
    if(cv2.waitKey(0) & 0xFF == ord("q")):
        continue

cv2.destroyAllWindows()