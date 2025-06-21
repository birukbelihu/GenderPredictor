import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def main():
    training_data_path = "gender-classification-dataset/Training/"
    validation_data_path = "gender-classification-dataset/Validation/"
    image_size = (224, 224)
    batch_size = 64
    epochs = 5

    print(f"Training Subfolders: {os.listdir(training_data_path)}")
    print(f"Validation Subfolders: {os.listdir(validation_data_path)}")

    data_generator = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    training_data = data_generator.flow_from_directory(
        training_data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    validation_data = data_generator.flow_from_directory(
        validation_data_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    print("Class indices:", training_data.class_indices)

    mobilenet_v2 = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    mobilenet_v2.trainable = False

    x = mobilenet_v2.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation="softmax")(x)

    gender_classification_model = Model(
        inputs=mobilenet_v2.input,
        outputs=output
    )
    gender_classification_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    gender_classification_model.summary()

    history = gender_classification_model.fit(
        training_data,
        validation_data=validation_data,
        epochs=epochs
    )

    loss, accuracy = gender_classification_model.evaluate(validation_data)
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)

    model_save_path = "gender_predictor.h5"
    gender_classification_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()