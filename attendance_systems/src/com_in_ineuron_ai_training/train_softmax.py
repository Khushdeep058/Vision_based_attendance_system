from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Import custom SoftMax and logger
from src.com_in_ineuron_ai_detectfaces_mtcnn.Configurations import get_logger
from src.com_in_ineuron_ai_training.softmax import SoftMax


class TrainFaceRecogModel:

    def __init__(self, args):
        self.args = args
        self.logger = get_logger()

        # Load the face embeddings
        self.data = pickle.loads(open(args["embeddings"], "rb").read())

    def trainKerasModelForFaceRecognition(self):
        # Encode the labels to integers
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        num_classes = len(np.unique(labels))

        # Convert embeddings to numpy array
        embeddings = np.array(self.data["embeddings"])

        # Initialize model parameters
        BATCH_SIZE = 8
        EPOCHS = 5
        input_shape = embeddings.shape[1]

        # Build Softmax classifier model
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # ✅ Compile model with sparse categorical crossentropy
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        # K-Fold Cross Validation
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        # Training loop
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val = embeddings[train_idx], embeddings[valid_idx]
            y_train, y_val = labels[train_idx], labels[valid_idx]

            # Fit model
            his = model.fit(
                X_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=1,
                validation_data=(X_val, y_val)
            )

            # ✅ Handle version differences for metric keys
            acc_key = "accuracy" if "accuracy" in his.history else "acc"
            val_acc_key = "val_accuracy" if "val_accuracy" in his.history else "val_acc"

            # Store metrics safely
            history['acc'] += his.history.get(acc_key, [])
            history['val_acc'] += his.history.get(val_acc_key, [])
            history['loss'] += his.history.get("loss", [])
            history['val_loss'] += his.history.get("val_loss", [])

            self.logger.info(f"Fold Accuracy: {his.history.get(acc_key, [])}")

        # ✅ Save the trained model
        model.save(self.args['model'])

        # ✅ Save the label encoder
        with open(self.args["le"], "wb") as f:
            pickle.dump(le, f)

        # Optional: Plot training metrics
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.legend()
        plt.title('Model Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title('Model Loss')

        plt.tight_layout()
        plt.savefig("training_results.png")
        plt.close()

        self.logger.info("Training complete and model saved successfully.")
