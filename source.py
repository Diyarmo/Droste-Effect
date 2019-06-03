import cv2
import numpy as np


image = cv2.imread("clock.jpg")
image = np.array(image)
rows = np.linspace(-1, 1, num=image.shape[0])
cols = np.linspace(-1, 1, num=image.shape[1])
xv, yv = np.meshgrid(rows, cols)
zv = xv + 1j*yv

#### Logarithmic Mapping
r1 = 0.2
r2 = 0.9
zv = zv.flatten()
zv = np.select([np.abs(zv) <= r2], [zv])
zv = np.select([np.abs(zv) >= r1], [np.log(zv/r1)])
zv = zv.reshape(xv.shape)
#### Logarithmic Mapping

'''
#### Rotation
zv = zv * (np.power(np.e, 1j*np.pi/4))
####Rotation
'''
rep = 2
Wx, Wy = np.real(zv), np.imag(zv)
Xnew = (Wx/np.max(np.abs(Wx)) + 1)*image.shape[1]/2
Ynew = (Wy/np.max(np.abs(Wy)) + 1)*image.shape[0]/2
Xnew = np.clip(Xnew, 0, image.shape[1]-1)
Ynew = np.clip(Ynew, 0, image.shape[0]-1)
Xnew = np.floor(Xnew).astype(int)
Ynew = np.floor(Ynew).astype(int)
Y = Ynew
for i in range(rep-1):
    Y = np.concatenate((Y, Ynew+(i+1)*image.shape[0]), axis=0)
Ynew = Y
Xnew = np.tile(Xnew, (rep, 1))
#alpha = np.arctan(np.log(r2/r1)/(2*np.pi))
#Z = Xnew + 1j*Ynew
#Z = Z * (np.power(np.e, 1j*alpha)) * np.cos(alpha)
#Xnew = np.real(Z)
#Ynew = np.imag(Z)
#Z = np.power(np.e, Z)

new_img = np.zeros([rep*image.shape[0], image.shape[1], 3], dtype=np.uint8)
for i in range(image.shape[0]):
    for j in range(rep*image.shape[1]):
        new_img[Ynew[j][i], Xnew[j][i]] = image[j % image.shape[1]][i]
cv2.imshow('image', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
