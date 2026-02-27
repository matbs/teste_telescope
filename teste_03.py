import cv2
import numpy as np
from picamera2 import Picamera2

def main():
    picam2 = Picamera2()
    
    # Configuração de vídeo
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    # --- O PULO DO GATO: CONTROLE DE EXPOSIÇÃO ---
    # Travamos a câmera para que ela fique "escura". 
    # Isso elimina ruídos e reflexos.
    picam2.set_controls({
        "AeEnable": False,           # Desliga o brilho automático
        "ExposureTime": 5000,        # 5ms (ajuste se a Lua estiver escura demais)
        "AnalogueGain": 1.0          # Ganho baixo para evitar ruído (granulação)
    })

    print("[INFO] Rastreio Lunar Robusto Iniciado...")

    try:
        while True:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Suaviza a imagem para ajudar na detecção de bordas
            gray = cv2.medianBlur(gray, 5)

            # Algoritmo de Hough Circles: Procura por círculos na imagem
            # dp=1.2, minDist=100 (distância entre duas luas)
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1.2, 
                minDist=100,
                param1=50,   # Sensibilidade de borda
                param2=30,   # Limiar de detecção (maior = mais rigoroso)
                minRadius=10, # Tamanho mínimo da lua no frame
                maxRadius=200 # Tamanho máximo
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    radius = i[2]
                    
                    # Desenha o contorno da Lua (Verde)
                    cv2.circle(frame, center, radius, (0, 255, 0), 3)
                    # Desenha o centro (Vermelho)
                    cv2.circle(frame, center, 2, (0, 0, 255), 3)
                    
                    cv2.putText(frame, f"LUA DETECTADA: {center}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Astro-Tracking Robusto", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
