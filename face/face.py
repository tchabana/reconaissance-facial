import cv2

# Récupérer le flux vidéo
cap = cv2.VideoCapture("4747")

# Tester le flux vidéo
if (cap.isOpened()):
    print("Impossible d'ouvrir le flux vidéo")
"""
Les lignes 13, 14, 33 et 42  servent à enregistrer le flux de la cémara et des détections faites
"""
# Définir le codec et créer l'objet VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Lecture des images
    _,image=cap.read()

    # Conversion a niveau de gris
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # detection des visages
    faces=face.detectMultiScale(image_gray,1.3,5)

    #for every face draw a rectangle
    for x,y, width,height in faces:
        cv2.rectangle(image,(x,y),(x+width,y+height),color=(255,0,0),thickness=1)
        
    # Enregistrer la trame d'image
    out.write(image)

    cv2.imshow('Camebush - Face detection', image)

    if cv2.waitKey(1)==ord('q'):
        break

# on libere les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
