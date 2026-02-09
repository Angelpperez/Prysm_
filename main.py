from vision_app.vision import BaslerUsbCamera, CameraCfg

import cv2 

import numpy as np 

import csv 

import time 

import math 

 
def main():
    # Parámetros de calibración (ejemplo: 1 píxel = 0.5 mm, 1 píxel= 1 mm) 

    MM_PER_PIXEL = 1 

    DISTANCIA_NOMINAL_MM = 1000 # (1 mts = 1000 mm) 

 

    # Inicializar cámara 

    video_path = "Muestra_01.mp4" 

    cap = cv2.VideoCapture(video_path) 

    #cap = cv2.VideoCapture(0) 

 

 

 

    # Archivo CSV 

    csv_file = open("distancias.csv", mode="w", newline="") 

    csv_writer = csv.writer(csv_file) 

    csv_writer.writerow(["Timestamp", "Distancia_px", "Distancia_mm", "Estado"]) 

 

    def calcular_centroide(contorno): 

        M = cv2.moments(contorno) 

        if M["m00"] != 0: 

            cx = int(M["m10"] / M["m00"]) 

            cy = int(M["m01"] / M["m00"]) 

            return (cx, cy) 

        return None 

 

    while True: 

        _, frame = cap.read() 

      
 

        # Convertir a gris y binarizar (ejemplo simple, ajustar según marcas reales) 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) 

 

        # Detectar contornos 

        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

 

        centros = [] 

        for cnt in contornos: 

            if cv2.contourArea(cnt) > 100:  # filtrar ruido 

                x, y, w, h = cv2.boundingRect(cnt) 

                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2) 

                centro = calcular_centroide(cnt) 

                if centro: 

                    centros.append(centro) 

                    cv2.circle(frame, centro, 5, (0,255,0), -1) 

 

        # Si hay al menos dos marcas 

        if len(centros) >= 2: 

            c1, c2 = centros[0], centros[1] 

            distancia_px = math.dist(c1, c2) 

            distancia_mm = distancia_px * MM_PER_PIXEL 

 

            # Estado 

            if abs(distancia_mm - DISTANCIA_NOMINAL_MM) <= 10:  # tolerancia ±10 mm 

                estado = "NORMAL" 

                color = (0,255,0) 

            else: 

                estado = "ALERTA" 

                color = (0,0,255) 

                # Aquí se puede enviar señal al PLC (ejemplo: Modbus/TCP o GPIO) 

 

            # Mostrar en pantalla 

            cv2.line(frame, c1, c2, color, 2) 

            cv2.putText(frame, f"{distancia_mm:.1f} mm ({estado})", (50,50), 

                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2) 

 

            # Guardar en CSV 

            csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), 

                                 distancia_px, distancia_mm, estado]) 

 

        cv2.imshow("Medicion", frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'): 

            break 

 

    cap.release() 

    csv_file.close() 

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()