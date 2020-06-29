import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../coverdata/20151204093048511..jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions, boxes = mser.detectRegions(gray)
print(img.shape)
print(len(boxes))
for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
print(img.shape)
plt.imshow(img, 'brg')
plt.show()
