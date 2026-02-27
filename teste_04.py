import cv2
import imutils
from picamera2 import Picamera2
import time

def main():
    # Inicializa a câmera
    picam2 = Picamera2()
    # Usamos uma resolução menor para garantir que o CSRT rode fluido na CPU
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    # Criamos o rastreador CSRT (O mais robusto do OpenCV)
    # Tentamos a API moderna e a legada para evitar erros no Trixie
    try:
        tracker = cv2.TrackerCSRT.create()
    except AttributeError:
        tracker = cv2.legacy.TrackerCSRT_create()

    initBB = None
    print("[INFO] Sistema pronto. Pressione 'S' para selecionar o objeto.")

    try:
        while True:
            frame = picam2.capture_array()
            # Redimensionar é vital para o CSRT não travar a Pi
            frame = imutils.resize(frame, width=450)
            (H, W) = frame.shape[:2]

            if initBB is not None:
                # Atualiza o rastreador
                (success, box) = tracker.update(frame)

                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Rastreando", (10, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "ALVO PERDIDO", (10, 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Tracking Robusto (Sem IA)", frame)
            key = cv2.waitKey(1) & 0xFF

            # Tecla 'S' para selecionar o objeto
            if key == ord("s"):
                # Seleciona a Região de Interesse (ROI)
                initBB = cv2.selectROI("Tracking Robusto (Sem IA)", frame, 
                                        fromCenter=False, showCrosshair=True)
                
                # Reinicializa o tracker com a nova seleção
                try:
                    tracker = cv2.TrackerCSRT.create()
                except AttributeError:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    
                tracker.init(frame, initBB)

            # Tecla 'Q' para sair
            elif key == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
