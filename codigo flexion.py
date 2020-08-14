# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def circulos(img):
    #Encuentra los circulos presentes en la imagen 
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,8,param1=50,param2=10,minRadius=10,maxRadius=18)
    return circles
    
    
def buscar_cordenadas(frame):
    cx=0
    cy=0
    #Se busca los contornos y se los dibuja
    contours,_ = cv2.findContours(frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #Buscamos el centro de los colores
    for i in contours:
        #Calcular el centro a partir de los momentos
        momentos = cv2.moments(i)
        cx = int(momentos['m10']/momentos['m00'])
        cy = int(momentos['m01']/momentos['m00'])
    return [cx,cy]

     
#########
#definimos el rango del color rojo en hsv
redBajo1 = np.array([0, 120, 120], np.uint8)
redAlto1 = np.array([1, 255, 255], np.uint8)
redBajo2 = np.array([173, 80, 100], np.uint8)
redAlto2 = np.array([180, 255, 255], np.uint8)

#definimos el rango del color azul en hsv
bluebajo=np.array([98,90,100],np.uint8)
bluealto=np.array([115,255,255],np.uint8)
#definimos el rango del color amarillo en hsv
yelbajo2=np.array([30,80,100],np.uint8)
yelalto2=np.array([80,255,255],np.uint8)

kernel = np.ones((3,3),np.uint8)# definimos matriz de unos para realizar transformaciones morfologicas
kernel1 = np.ones((9,9),np.uint8)# definimos matriz de unos para realizar transformaciones morfologicas

##variables
contador_frames=0
angulo=0
angulo1=0
angulos=[]
velocidades=[]
bandera=0;
periodo=0;
periodos=[];



###########################
video=cv2.VideoCapture('C:/Users/USWER/Desktop/Universidad/Trabajo de grado/100CANON/prueba21.mp4') # se abre el video como objeto 

while True:
    std, frame=video.read() #se recibe un booleano y el fotograma al leer el objeto creado  
    #condicion para comprbar que el video termino
    if std!=True:# si el booleano es falso entonces se termina el ciclo
        print('No hay fotograma')
        break 
    
    frame=cv2.resize(frame,(640,480)) # se hace una reduccion del fotograma
    frame1 = cv2.GaussianBlur(frame, (17,17), cv2.BORDER_DEFAULT)#se suavisa el fotograma
    frame_hsv=cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)# se convierte la imagen a hsv

    
    ####se realiza procesamiento del color rojo#####
    maskRed1 = cv2.inRange(frame_hsv, redBajo1, redAlto1)#se hace el fitro con el primer rango de rojo
    maskRed2 = cv2.inRange(frame_hsv, redBajo2, redAlto2)#se hace el fitro con el segundo rango de rojo
    maskRed = cv2.add(maskRed1, maskRed2)# se unen los dos filtros de rojo 
    img_cr_red=cv2.morphologyEx(maskRed, cv2.MORPH_CLOSE, kernel)# se realiza un cierre para poder definir mejor los bordes
    img_di_red=cv2.dilate(img_cr_red,kernel,iterations=1)# se realiza una dilatacion de la imagen
    cor_rojo=buscar_cordenadas(img_di_red)
    #print("cordenadas rojo: ", cor_rojo)
    ###############
    #### se realiza procesamiento para el color amarillo.#####
    maskyel = cv2.inRange(frame_hsv, yelbajo2, yelalto2)#se hace el fitro con el primer rango de amarillo
    img_cr_yel=cv2.morphologyEx(maskyel, cv2.MORPH_CLOSE, kernel)# se realiza un cierre para poder definir mejor los bordes
    img_di_yel=cv2.dilate(img_cr_yel,kernel,iterations=1)# se realiza una dilatacion de la imagen
    cor_amarillo=buscar_cordenadas(img_di_yel)
    #print("cordenadas amarillo: ", cor_amarillo)
    #####
    #### se realiza procesamiento para el color azul.#####
    maskblue = cv2.inRange(frame_hsv, bluebajo, bluealto)#se hace el fitro con el primer rango de azul
    img_cr_blue=cv2.morphologyEx(maskblue, cv2.MORPH_CLOSE, kernel)# se realiza un cierre para poder definir mejor los bordes
    img_di_blue=cv2.dilate(img_cr_blue,kernel,iterations=1)# se realiza una dilatacion de la imagen 
    cor_azul=buscar_cordenadas(img_di_blue)
    #print("cordenadas azul: ", cor_azul)
    #######
    #se une los colores filtrados es una sola imagen.
    mask = cv2.add(img_di_red, img_di_yel)
    mask = cv2.add(mask, img_di_blue)
    
    ##se dibuja los puntos encontrados
    cv2.circle(frame,(cor_rojo[0], cor_rojo[1]), 4, (0,255,0), -1)#rojo
    cv2.circle(frame,(cor_azul[0], cor_azul[1]), 4, (0,255,0), -1)#azul
    cv2.circle(frame,(cor_amarillo[0], cor_amarillo[1]), 4, (0,255,0), -1)#amarillo
    
    ##aseguramos que todos los segmentos se reconoscan en la imagen 
    if (cor_rojo[0]!=0 and cor_rojo[1]!=0 and cor_azul[0]!=0 and cor_azul[1]!=0 and cor_amarillo[0]!=0 and cor_amarillo[1]!=0):
        if angulo!=0:
            angulo1=angulo#guardamos el angulo en otra variable para poder hacer calculo de velocidad angular    
        contador_frames+=1
        ##Calculamos los segmentos
        RA=math.sqrt(((cor_rojo[0]-cor_azul[0])**2)+((cor_rojo[1]-cor_azul[1])**2))
        AAM=math.sqrt(((cor_amarillo[0]-cor_azul[0])**2)+((cor_amarillo[1]-cor_azul[1])**2))
        AMR=math.sqrt(((cor_rojo[0]-cor_amarillo[0])**2)+((cor_rojo[1]-cor_amarillo[1])**2))
        #calculamos el angulo
        angulo=math.acos(((RA**2)+(AAM**2)-(AMR**2))/(2*RA*AAM))#Angulo en radianes
        angulo=math.degrees(angulo)#angulo en grados
        cv2.putText(frame,str(angulo),(cor_azul[0]+10,cor_azul[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
        angulos.append(angulo)
        if angulo1!=0:
            velocidad=(abs(angulo1-angulo))*30#se calcula la velocidad por fotogramas
            velocidades.append(velocidad)
        if (angulo<=50):
            bandera=1
        if ((angulo>=160) and (bandera==1) and (angulo<angulo1)):
            # print("ciclo")
            # print(angulo,angulo1)
            periodo=float(contador_frames)/(30/1000)#calculamos el periodo en milisedundos 
            periodos.append(periodo)#se agrega a la lista para poder graficar 
            # reiniciamos variables 
            bandera=0;
            contador_frames=0;
        # print("angulo= ",angulo)
        # print("angulo1= ",angulo1)
        # print (contador_frames)
            
    
    
    
    # #Se busca los contornos y se los dibuja
    # contours,_ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame, contours, -1, (0,255,0), 2)
    
    # #Buscamos el centro de los colores
    # for i in contours:
    # #Calcular el centro a partir de los momentos
    #     momentos = cv2.moments(i)
    #     cx = int(momentos['m10']/momentos['m00'])
    #     cy = int(momentos['m01']/momentos['m00'])
    #     #Dibujar el centro
    #     cv2.circle(frame,(cx, cy), 3, (0,255,0), -1)

    
    
    # cnts= cv2.findContours(img_ap_red, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] #Encontar contornos
    # for c in cnts:
    #     (x,y,w,h)=cv2.boundingRect(c)
    #     recorte=frame2[y:h+y,x:w+x]
    #     recorte3=np.float32(recorte)
    #     recorte1=frame[y:h+y,x:w+x]
    #     dms=cv2.cornerHarris(recorte3,2,3,0.04)
    #     
    #     dms = cv2.dilate(dms, None) 
    #     recorte1[dms > 0.01 * dms.max()]=[0, 0, 255] 
    #     cv2.imshow('recorte',recorte1)
          
        
        
        
    
    
    # cir_red=circulos(img_cr_red)
    # 
    #   
    # if cir_red is not None:
    #     circles = np.round(cir_red[0, :]).astype("int")
    #     print (circles) 
    #     for (x0, y0, r0) in circles:
    #         cv2.circle(frame, (x0, y0), r0,(255,0,0),2)
    #         print ("coordinates", x0, y0, r0)
    # else:
    #     print('Blink')
     
    cv2.imshow('ventana1',frame)# se muestra el video
    cv2.imshow('ventana4',mask)# se muestra el video
    if cv2.waitKey(5) & 0xFF == ord('q'): #se presiona q para poder salir del video
        break
 

##Procesamiento de resultados
#grafricamos angulos
plt.figure()
plt.plot(angulos)
plt.title("ANGULO (°)")
plt.show()

# #graficamos velocidades
plt.figure()
plt.plot(velocidades)
plt.title("velocidad (°/s)")
plt.show()

#graficamos periodos de ciclos
plt.figure()
plt.plot(periodos)
plt.title("periodo(ms)")
plt.show()

cv2.destroyAllWindows() #destruye todas las ventanas creadas en el codigo
    

