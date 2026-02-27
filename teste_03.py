import cv2
import imutils
from ultralytics import YOLO
from picamera2 import Picamera2

# Variáveis globais para o clique
selected_id = None

def select_object(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        results = param # Recebe os resultados do YOLO
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # Verifica se o clique foi dentro de alguma caixa
                if x1 <= x <= x2 and y1 <= y <= y2:
                    selected_id = int(ids[i])
                    print(f"[INFO] Objeto travado! ID: {selected_id}")
                    break

def main():
    global selected_id
    model = YOLO('yolov8n.pt') 
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()

    cv2.namedWindow("YOLOv8 Click Tracking")
    
    print("[DICA] Clique em um objeto na tela para focar nele.")
    
    try:
        results = None
        while True:
            frame = picam2.capture_array()
            frame = imutils.resize(frame, width=640)

            # Define a função de clique e passa os resultados atuais
            cv2.setMouseCallback("YOLOv8 Click Tracking", select_object, param=results)

            results = model.track(frame, persist=True, conf=0.3, verbose=False)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    obj_id = int(ids[i])
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Se não selecionamos nada, mostra todos. 
                    # Se selecionamos, destaca apenas o escolhido.
                    if selected_id is None or obj_id == selected_id:
                        color = (0, 255, 0) if obj_id == selected_id else (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("YOLOv8 Click Tracking", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            if key == ord("r"): selected_id = None # 'r' para resetar a seleção

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
