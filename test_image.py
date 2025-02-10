import cv2
from ultralytics import YOLO
import os

# Carregue o modelo (pode ser yolov8n.pt, yolov8s.pt, etc.)
model = YOLO('runs/detect/train8/weights/best.pt')

# Defina uma lista de imagens (pode ser só uma ou várias)
imagens = [
    "train/images/img1.jpg",
    "train/images/img2.jpg",
    # ...adicione mais caminhos se quiser
]

# Diretório onde as imagens resultantes serão salvas (opcional)
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

for img_path in imagens:
    # Faz a predição
    results = model.predict(img_path, conf=0.5)  # conf: threshold de confiança

    result = results[0]

    # O `result.plot()` gera um ndarray do frame já anotado
    annotated_image = result.plot()

    # Mostrar na tela (opcional)
    cv2.imshow("Detecções", annotated_image)
    cv2.waitKey(0)  # aguarda até fechar a janela

    # Salvar em arquivo (opcional)
    # Ex: "foto1.jpg" => "results/foto1_result.jpg"
    base_name = os.path.basename(img_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(save_dir, f"{name_without_ext}_result.jpg")
    cv2.imwrite(output_path, annotated_image)
    print(f"Imagem anotada salva em: {output_path}")

cv2.destroyAllWindows()
