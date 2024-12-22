import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tkinter import Tk, filedialog, Button, Label, messagebox

# Пути и параметры
CUSTOM_MODEL_PATH = r'C:\Users\maksc\OneDrive\Рабочий стол\Идз\final_animal_recognition_model.h5'
IMAGE_SIZE = (224, 224)  # Размер изображений для обработки

# Загрузка моделей
try:
    custom_model = load_model(CUSTOM_MODEL_PATH)
    print(f"Кастомная модель загружена из {CUSTOM_MODEL_PATH}")
except Exception as e:
    print(f"Ошибка при загрузке кастомной модели: {e}")
    custom_model = None

mobilenet_model = MobileNetV2(weights="imagenet")
print("Модель MobileNetV2 загружена.")

# Классы кастомной модели
custom_class_labels = {
    0: 'cat',
    1: 'dog',
    2: 'panda',
    3: 'unknown'
}

# Предварительная обработка изображения для обеих моделей
def preprocess_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = image / 255.0  # Нормализация
    return np.expand_dims(image, axis=0)

def preprocess_image_mobilenet(image):
    image = cv2.resize(image, (224, 224))
    return preprocess_input(np.expand_dims(image, axis=0))

# Функция предсказания для кастомной модели
def predict_with_custom_model(image, threshold=0.6):
    processed_image = preprocess_image(image)
    predictions = custom_model.predict(processed_image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    if confidence < threshold:
        return "unknown", confidence
    return custom_class_labels[class_id], confidence

# Функция предсказания для MobileNetV2
def predict_with_mobilenet(image):
    processed_image = preprocess_image_mobilenet(image)
    preds = mobilenet_model.predict(processed_image)
    decoded = decode_predictions(preds, top=1)[0]
    label, confidence = decoded[0][1], decoded[0][2]
    return label, confidence

# Работа с камерой
def recognize_with_camera(model_choice):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Ошибка", "Не удалось открыть камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if model_choice == "custom":
            label, confidence = predict_with_custom_model(frame)
        else:
            label, confidence = predict_with_mobilenet(frame)

        display_text = f"{label}: {confidence:.2f}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Recognition (Camera)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Работа с загруженным изображением
def recognize_from_file(model_choice):
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        messagebox.showinfo("Информация", "Файл не выбран.")
        return

    try:
        # Загрузка изображения с помощью PIL
        image = Image.open(file_path).convert('RGB')  # Конвертируем в RGB
        image = np.array(image)

        if model_choice == "custom":
            label, confidence = predict_with_custom_model(image)
        else:
            label, confidence = predict_with_mobilenet(image)

        messagebox.showinfo("Результат", f"Класс: {label}\nУверенность: {confidence:.2f}")
    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при обработке изображения: {e}")

# Графический интерфейс
def create_gui():
    root = Tk()
    root.title("Object Recognition")
    root.geometry("300x300")

    Label(root, text="Выберите модель:", font=("Arial", 14)).pack(pady=10)

    Button(root, text="Кастомная модель: Фото", command=lambda: recognize_from_file("custom"), width=25, height=2).pack(pady=10)
    Button(root, text="Кастомная модель: Камера", command=lambda: recognize_with_camera("custom"), width=25, height=2).pack(pady=10)

    Button(root, text="MobileNetV2: Фото", command=lambda: recognize_from_file("mobilenet"), width=25, height=2).pack(pady=10)
    Button(root, text="MobileNetV2: Камера", command=lambda: recognize_with_camera("mobilenet"), width=25, height=2).pack(pady=10)

    root.mainloop()

# Запуск приложения
if __name__ == "__main__":
    create_gui()