# final-exam

700766438   singamshetty johnson
video link: https://drive.google.com/drive/folders/1eZgmEJFvR8a6TB0Cdxty2LDSg0J58MSH?usp=sharing


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Model builder function
def build_model(optimizer='adam'):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 3. Manual hyperparameter search
optimizers = ['adam', 'sgd']
batch_sizes = [64, 128]
best_acc = 0
best_params = {}
for opt in optimizers:
    for bs in batch_sizes:
        model = build_model(optimizer=opt)
        history = model.fit(x_train, y_train_cat, epochs=3, batch_size=bs, verbose=0, validation_data=(x_test, y_test_cat))
        acc = history.history['val_accuracy'][-1]
        print(f"Optimizer: {opt}, Batch size: {bs}, Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = {'optimizer': opt, 'batch_size': bs}

print("Best params:", best_params)

# 4. Train final model with best hyperparameters and ReduceLROnPlateau
final_model = build_model(optimizer=best_params['optimizer'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
history = final_model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=best_params['batch_size'],
    validation_data=(x_test, y_test_cat),
    callbacks=[reduce_lr]
)

# 5. Confusion matrix
y_pred = np.argmax(final_model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 6. Training/testing loss and accuracy plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.legend()
plt.title('Accuracy')
plt.show()

# 7. ROC curve for one class (e.g., digit 0)
y_score = final_model.predict(x_test)
fpr, tpr, _ = roc_curve(y_test_cat[:, 0], y_score[:, 0])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Digit 0')
plt.legend()
plt.show()
