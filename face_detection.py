import cv2

# Carregar o modelo pré-treinado para detecção de faces do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Checar se a Webcam foi aberta corretamente
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Ler um quadro do vídeo
    ret, frame = cap.read()

    # Converter o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Correção aqui

    # Detectar faces no quadro
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Desenhar retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Exibir o quadro resultante
    cv2.imshow('Face Detection', frame)

    # Parar o loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()