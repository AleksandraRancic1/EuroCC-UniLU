"""
Exercise 1_tf: we will design few CNN models and run them on HPC.
The dataset we will use os CIFAR-10, and it is supported in tensorflow library.
This dataset requires minimal preprocessing. Note that this is not the
case with all the other datasets. 
Steps:
1. Import libraries
2. load the dataset
3. normalize the dataset
4. introduce callback function - track metrics with timing
5. Create models
6. Save metrics functions
7. Run models
8. Comment on metrics and the influence of different parameters
"""

# 1. Import libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sys
import time
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

output_dir = "/home/users/arancic/EuroCC/outputs_ex1"
os.makedirs(output_dir, exist_ok=True)  # this should be changed for meluxina

# 2. Load the dataset -> CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 3. Normalize the dataset
"""
Pixel values in CIFAR-10 are in the range [0, 255]. Since this scale is
not ideal for training deep neural networks, we need to normalize the dataset.
The normalization is done dividing by 255.0, and on that way all pixel values are
rescaled to the range [0, 1]. This is especially important when using gradient-based optimizers.
"""
x_train, x_test = x_train / 255.0, x_test / 255.0

# 4. Track metrics

class TimingLogger(Callback):
    def on_train_begin(self, logs=None):
        self.times = []
        self.history = {'val_accuracy': [], 'val_loss': [], 'accuracy': [], 'loss': []}
        self.best_val_acc = 0
        self.best_epoch = -1

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        self.times.append(duration)
        for k in self.history:
            self.history[k].append(logs.get(k))
        print(f"Epoch {epoch+1} — acc: {logs['accuracy']:.4f}, loss: {logs['loss']:.4f}, "
              f"val_acc: {logs['val_accuracy']:.4f}, val_loss: {logs['val_loss']:.4f}, "
              f"time: {duration:.2f}s")
        if logs["val_accuracy"] > self.best_val_acc:
            self.best_val_acc = logs["val_accuracy"]
            self.best_epoch = epoch


# 5. Create models

model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model3 = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model4 = models.Sequential([

    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')

])

# 6. Save metrics functions

def save_metrics_csv(callback, model_name):
    df = pd.DataFrame(callback.history)
    df['epoch_time'] = callback.times
    df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)

def save_best_summary(model_name, callback, y_true, y_pred):
    best_epoch = callback.best_epoch
    f1 = f1_score(y_true, y_pred, average='macro')
    with open(os.path.join(output_dir,"summary_results.txt"), "a") as f:
        f.write(f"{model_name}: best val_ac = {callback.best_val_acc:.4f} at epoch {best_epoch+1}, F1 = {f1:.4f}\n")

def plot_metric_across_models(metric, histories, labels, filename):
    plt.figure()
    for h, label in zip(histories, labels):
        plt.plot(h.history[metric], label=label)
    plt.title(f"{metric.replace('_', ' ').title()} per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.savefig(filename)

def plot_confusion_matrices(model_preds, labels_true, model_names, filename):
    fig, axes = plt.subplots(1, len(model_preds), figsize=(6 * len(model_preds), 5))
    for i, (y_pred, name) in enumerate(zip(model_preds, model_names)):
        cm = confusion_matrix(labels_true, y_pred)
        disp=ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[i], colorbar=False)
        axes[i].title.set_text(name)
    plt.tight_layout()
    plt.savefig(filename)

# 7. Compile the models and run them

# Model 1
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.summary()
sys.stdout.flush()
logger1 = TimingLogger()
model1.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[logger1])
save_metrics_csv(logger1, "model1")
y_pred1 = model1.predict(x_test).argmax(axis=1)
save_best_summary("model1", logger1, y_test.flatten(), y_pred1)

# Model 2
optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-4)
model2.compile(optimizer=optimizer2, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.summary()
sys.stdout.flush()
logger2 = TimingLogger()
model2.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), callbacks=[logger2])
save_metrics_csv(logger2, "model2")
y_pred2 = model2.predict(x_test).argmax(axis=1)
save_best_summary("model2", logger2, y_test.flatten(), y_pred2)

# Model 3
optimizer3 = tf.keras.optimizers.Adam(learning_rate=1e-4)
model3.compile(optimizer=optimizer3, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.summary()
sys.stdout.flush()
logger3 = TimingLogger()
model3.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), callbacks=[logger3])
save_metrics_csv(logger3, "model3")
y_pred3 = model3.predict(x_test).argmax(axis=1)
save_best_summary("model3", logger3, y_test.flatten(), y_pred3)

# Model 4
optimizer4 = tf.keras.optimizers.Adam(learning_rate=1e-4)
model4.compile(optimizer=optimizer4, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model4.summary()
sys.stdout.flush()
logger4 = TimingLogger()
model4.fit(x_train, y_train, batch_size=64, epochs=50,
          validation_data=(x_test, y_test),
          callbacks=[
              tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
              tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
              logger4
          ])
save_metrics_csv(logger4, "model4")
y_pred4 = model4.predict(x_test).argmax(axis=1)
save_best_summary("model4", logger4, y_test.flatten(), y_pred4)

# 8. Plots

plot_metric_across_models("val_accuracy",
    [logger1, logger2, logger3, logger4],
    ["model1", "model2", "model3", "model4"],
    filename=os.path.join(output_dir, "val_accuracy_all_models.png"))

plot_metric_across_models("val_loss",
    [logger1, logger2, logger3, logger4],
    ["model1", "model2", "model3", "model4"],
    filename=os.path.join(output_dir, "val_loss_all_models.png"))

plot_confusion_matrices(
    [y_pred1, y_pred2, y_pred3, y_pred4],
    y_test.flatten(),
    ["model1", "model2", "model3", "model4"],
    filename = os.path.join(output_dir, "confusion_matrices.png")
)


# Predictions

def save_prediction_grid(images, y_true, y_pred, class_names, filename, n=25):
    plt.figure(figsize=(12, 12))
    indices = np.random.choice(len(images), n, replace=False)
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[idx])
        plt.axis('off')
        true_label = class_names[y_true[idx][0]]
        pred_label = class_names[y_pred[idx]]
        if pred_label == true_label:
            color = "green"
            label = f"✓ {pred_label}"
        else:
            color = "red"
            label = f"✗ {pred_label}\n({true_label})"
        plt.title(label, color=color, fontsize=8)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

save_prediction_grid(x_test, y_test, y_pred1, class_names, os.path.join(output_dir, "model1_predictions_grid.png"))
save_prediction_grid(x_test, y_test, y_pred2, class_names, os.path.join(output_dir, "model2_predictions_grid.png"))
save_prediction_grid(x_test, y_test, y_pred3, class_names, os.path.join(output_dir, "model3_predictions_grid.png"))
save_prediction_grid(x_test, y_test, y_pred4, class_names, os.path.join(output_dir, "model4_predictions_grid.png"))