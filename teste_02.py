import cv2
import imutils
import time
import numpy as np
from imutils.video import FPS
from picamera2 import Picamera2

# -------------------------------------------------------
# Inicialização da Pi Camera
# -------------------------------------------------------
def start_camera():
    picam2 = Picamera2()
    
    # Criamos uma configuração para captura de vídeo em formato BGR para o OpenCV
    # O Trixie lida melhor com streams explícitos
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    print("[INFO] Câmera aquecendo...")
    time.sleep(2.0)
    return picam2

# -------------------------------------------------------
# Criação do tracker
# -------------------------------------------------------
def create_tracker():
    # Nota: Em versões novas do OpenCV, os trackers ficam em cv2.legacy ou mudaram a API
    # TrackerKCF costuma ser mais rápido que MIL para Raspberry Pi
    return cv2.TrackerKCF_create()

# -------------------------------------------------------
# Loop de tracking
# -------------------------------------------------------
def run_tracking_loop(picam2):
    tracker = None
    initBB = None
    fps = None

    while True:
        # Captura o frame diretamente como um array numpy
        frame = picam2.capture_array()
        
        # Redimensionar ajuda na velocidade do processamento
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]

        if initBB is not None:
            # Tenta atualizar o tracker
            success, box = tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Rastreando", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Alvo Perdido", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            fps.update()
            fps.stop()

            cv2.putText(frame, f"FPS: {fps.fps():.2f}", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Pi Camera - Tracking", frame)
        key = cv2.waitKey(1) & 0xFF

        # Pressione 's' para selecionar ROI
        if key == ord("s"):
            # O selectROI precisa de uma janela limpa
            initBB = cv2.selectROI("Pi Camera - Tracking", frame, fromCenter=False, showCrosshair=True)
            
            # Reinicializa o tracker com a nova seleção
            tracker = create_tracker()
            tracker.init(frame, initBB)
            fps = FPS().start()

        # Pressione 'q' para sair
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    picam2.stop()

def main():
    print("[INFO] Iniciando sistema...")
    try:
        picam2 = start_camera()
        run_tracking_loop(picam2)
    except Exception as e:
        print(f"[ERRO] Falha ao iniciar: {e}")

if __name__ == "__main__":
    main()
