import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Параметры
DATASET_PATH = r'C:\Users\maksc\data\animals'  # Замените на путь к вашим данным
IMAGE_SIZE = (224, 224)  # MobileNetV2 требует размер 224x224
BATCH_SIZE = 32
EPOCHS = 10  # Настройте под ваши данные и ресурсы
LEARNING_RATE = 0.0001

# Проверка пути к данным
if not os.path.exists(DATASET_PATH):
    raise ValueError(f"Указанный путь к данным {DATASET_PATH} не существует.")

# Аугментация данных
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% для обучения, 20% для валидации
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Предобученная модель
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model.trainable = False  # Замораживаем базовые слои

# Кастомная голова модели
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Уменьшение переобучения
    Dense(train_generator.num_classes, activation='softmax')  # Количество классов (например, 4)
])

# Компиляция модели
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Коллбэки
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='final_animal_recognition_model.keras',  # Изменено на .keras
    monitor='val_loss',
    save_best_only=True
)

# Обучение модели
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint]
)

# Сохранение обученной модели
print(f"Модель сохранена по пути: final_animal_recognition_model.h5")
