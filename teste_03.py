import cv2
import imutils
import numpy as np
from picamera2 import Picamera2

def main():
    # 1. Inicialização da Picamera2
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    # 2. Configurações Manuais de Exposição (Crucial para a Lua)
    # Baixamos o tempo de exposição e o ganho para ver detalhes e não um borrão
    # Esses valores podem precisar de ajuste fino dependendo da fase da lua
    picam2.set_controls({
        "ExposureTime": 5000,  # Em microsegundos (ajuste entre 2000 e 10000)
        "AnalogueGain": 1.0,    # Ganho baixo para evitar ruído
        "AeEnable": False       # Desativa o auto-exposição
    })

    print("[INFO] Rastreio Lunar Otimizado Iniciado...")
    print("[INFO] Pressione 'q' para sair.")

    try:
        while True:
            # Captura o frame
            frame = picam2.capture_array()
            
            # Pré-processamento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 0)

            # Threshold adaptativo ou simples (ajuste o valor 200 conforme necessário)
            _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
            
            # Limpeza morfológica para remover ruídos menores
            mask = cv2.erode(thresh, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Encontra contornos
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if len(cnts) > 0:
                # Filtra pelo maior contorno
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                # Cálculo de Circularidade (Lua cheia/quarto crescente são formas definidas)
                # area_circulo_teorico = pi * r^2
                circularity = area / (np.pi * (radius ** 2))

                # Filtro: Apenas objetos com tamanho mínimo e relativamente redondos
                if radius > 10 and circularity > 0.4:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Desenha a marcação
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                        cv2.drawMarker(frame, (cX, cY), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
                        
                        # Info na tela
                        cv2.putText(frame, f"LUA DETECTADA", (cX - 50, int(y - radius - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"Coord: {cX}, {cY}", (10, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Exibição
            cv2.imshow("Monitoramento Lunar", frame)
            # cv2.imshow("Mascara", mask) # Debug

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("[INFO] Encerrando...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
