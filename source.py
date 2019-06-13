import cv2
import numpy as np
img = cv2.imread("clock.jpg")
rows = np.linspace(-1, 1, num=img.shape[0])
cols = np.linspace(-1, 1, num=img.shape[1])
xv, yv = np.meshgrid(cols, rows)
zv = xv + 1j*yv
r1, r2 = 0.2, 0.9
zv = zv.flatten()
zv = np.select([np.abs(zv) <= r2], [zv])
zv = np.select([np.abs(zv) >= r1], [np.log(zv/r1)])
zv = zv.reshape(xv.shape)
rep = 5
Znew = np.tile(zv, (rep, 1))
for i in range(rep):
    Znew[i*img.shape[0]:(i+1)*img.shape[0]] += 2j*i*np.max(np.abs(np.imag(zv)))
alpha = np.arctan(np.log(r2/r1)/(2*np.pi))
Znew = Znew * (np.power(np.e, 1j*alpha)) * np.cos(alpha)
Znew = np.power(np.e, Znew)
Xnew, Ynew = np.real(Znew), np.imag(Znew)
Xnew = (Xnew/np.max(np.abs(Xnew)) + 1)*img.shape[1]/2
Ynew = (Ynew/np.max(np.abs(Ynew)) + 1)*img.shape[0]/2
Xnew = np.clip(Xnew, 0, img.shape[1]-1)
Ynew = np.clip(Ynew, 0, img.shape[0]-1)
Xnew = np.floor(Xnew).astype(int)
Ynew = np.floor(Ynew).astype(int)
new_img = np.zeros([3*img.shape[0], img.shape[1], 3], dtype=np.uint8)
for i in range(rep*img.shape[0]):
    for j in range(img.shape[1]):
        new_img[Ynew[i][j], Xnew[i][j]] = img[i % img.shape[0]][j]
cv2.imwrite('out.jpg', new_img)
