

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
# YOLO ağırlıklarının ve konfigürasyon dosyasının yolu
yolo_weights_path = "yolov3.weights"

yolo_config_path = "yolov3.cfg"


# YOLO sınıf etiketlerinin yolu
yolo_labels_path = "coco.names"
LABELS = open(yolo_labels_path).read().strip().split("\n")

# YOLO'nun tespit eşiği
confidence_threshold = 0.5

# ResNet modelinin yolu
resnet_model_path = "Nuts-Classification-Rexnetr50x1.hdf5"

# YOLO ağı yükleniyor
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

# ResNet modeli yükleniyor


resnet_model = tf.keras.models.load_model(
       ("Nuts-Classification-Rexnetr50x1.hdf5"),
       custom_objects={'KerasLayer':hub.KerasLayer}
)


# ResNet modelinin giriş boyutları
input_width = 300
input_height = 300

# ResNet sınıf etiketleri
resnet_labels = ["Pistachios", "Sunflower Seeds", "Pumpkin Seeds", "Roasted Chickpea", "Peanut"]  # ResNet sınıf etiketlerini burada değiştirin

# Kamera bağlantısını başlat
cap = cv2.VideoCapture(0)

# YOLO ağının çıktı katmanlarını bulma fonksiyonu
def get_output_layers(net):
    layer_names = net.getLayerNames()
    
    outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return outputlayers

while True:
    # Kameradan bir kare al
    ret, frame = cap.read()

    if not ret:
        break

    # Görüntüyü YOLO için önceden işleme
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # YOLO ile nesne tespiti yap
    net.setInput(blob)
    outputs = net.forward(get_output_layers(net))

    # Tespit edilen nesneleri saklamak için listeler oluştur
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                # Nesne merkez koordinatlarını ve boyutlarını hesapla
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                box_width = int(detection[2] * frame.shape[1])
                box_height = int(detection[3] * frame.shape[0])

                # Koordinatlardan köşe noktalarını tespit et
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)

                boxes.append([x, y, box_width, box_height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maximum Suppression (NMS) uygula
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

    # Nesneleri tespit et ve çerçevele
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            # Görüntüyü ResNet için önceden işleme
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (input_width, input_height))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = np.expand_dims(roi, axis=0)

            # ResNet ile tahmin yap
            predictions = resnet_model.predict(roi)
            predicted_class = np.argmax(predictions)

            # Nesneyi çerçevele ve sınıf etiketini ekle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,str(round(np.max(predictions),2))+" "+ resnet_labels[predicted_class], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow("Object Detection", frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()