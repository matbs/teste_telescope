import cv2
import imutils
import numpy as np
from picamera2 import Picamera2

def main():
    # Inicializa a Picamera2
    picam2 = Picamera2()
    # Para a Lua, precisamos controlar a exposição (para não virar apenas um borrão branco)
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()

    # Ajustes de exposição para a Lua (evita que o brilho estoure)
    # Nota: No Trixie, estes controles são feitos via as propriedades da câmera
    # controls = {"ExposureTime": 10000, "AnalogueGain": 1.0} 
    # picam2.set_controls(controls)

    print("[INFO] Rastreio Lunar Iniciado...")
    print("[INFO] Pressione 'q' para sair.")

    try:
        while True:
            frame = picam2.capture_array()
            # Converte para tons de cinza para focar no brilho
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Aplica um desfoque para remover ruído digital
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)

            # THRESHOLD: O "pulo do gato". Isolamos apenas os pixels muito brilhantes.
            # Ajuste o valor 200 se a Lua estiver fraca (nuvens) ou forte.
            _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

            # Encontra os contornos da mancha branca (Lua)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if len(cnts) > 0:
                # Pega o maior contorno brilhante (presumidamente a Lua)
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                
                # Calcula o centro (Centroide)
                if M["m00"] > 0:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # Só desenha se o "objeto" tiver um tamanho mínimo
                    if radius > 5:
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"LUA: {center}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Rastreio Lunar - Pi Camera", frame)
            # cv2.imshow("Mascara de Brilho", thresh) # Descomente para ver o que a câmera "vê"

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
