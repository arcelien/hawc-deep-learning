import numpy as np
import matplotlib.pyplot as plt

""" To run:
	1. Load in saved numpy images (n, 40, 40, 2)
	2. Choose image number (1, 40, 40, 2)
	3. Create folder imgs
	4. Run code
	5.run 'ffmpeg -start_number 0 -i imgs/img%00d.jpg -vcodec mpeg4 test.mp4' in terminal
"""

image_num = 0

imgs = np.load("arr_0.npy")
img = imgs[image_num,:,:,:]

outarr = np.zeros((100, 40, 40))
outarr[:,39,0] = .1
a = img
a[:,:,1] += 1
a[:,:,1] *= 50
for i in range(a.shape[0]):
	for j in range(a.shape[1]):
		for alpha in range(0, 26):
			fade = (1-.0016*alpha**2)
			outarr[int(a[i, j, 1])+alpha:int(a[i, j, 1])+alpha+1, i, j] = int((a[i,j,0] + 1) * 127) * fade


for i in range(100):
	plt.figure()
	plt.imshow(outarr[i,:,:])
	plt.savefig("imgs/img"+str(i)+".jpg")
	plt.close()
