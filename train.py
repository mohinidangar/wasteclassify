# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
#
# # Paths
# train_dir = r'D:\pythonProject\pythonProject\example\dataset\train'
# val_dir = r'D:\pythonProject\pythonProject\example\dataset\validation'
#
# # Data generators
# train_datagen = ImageDataGenerator(rescale=1. / 255)
# val_datagen = ImageDataGenerator(rescale=1. / 255)
#
# train_gen = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     class_mode='categorical'
# )
# val_gen = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(150, 150),
#     class_mode='categorical'
# )
#
# # Get number of classes dynamically
# num_classes = len(train_gen.class_indices)
#
# # Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dense(num_classes, activation='softmax')
# ])
#
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Train model
# model.fit(train_gen, validation_data=val_gen, epochs=10)
#
# # Save model
# model.save('waste_classifier_model.h5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# === Paths ===
train_dir = r'D:\pythonProject\pythonProject\example\dataset\train'
val_dir = r'D:\pythonProject\pythonProject\example\dataset\validation'

# === Data Generators with Augmentation ===
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# === Get number of classes ===
num_classes = len(train_gen.class_indices)

# === Base Model: MobileNetV2 ===
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base

# === Custom Classification Head ===
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Compile Model ===
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model_mobilenetv2.h5', monitor='val_loss', save_best_only=True)
]

# === Train Model ===
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks
)

# === Optionally Fine-Tune Base Model ===
# Unfreeze some top layers of the base model and recompile
base_model.trainable = True
for layer in base_model.layers[:-30]:  # Freeze all except last 30 layers
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune with lower learning rate
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks
)

# === Save Final Model ===
model.save('waste_classifier_finetuned_mobilenetv2.h5')

