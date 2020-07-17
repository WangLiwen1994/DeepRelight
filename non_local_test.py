import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread("dataset/track3_HDR/Image198.png",cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(8,8))
    img_flat = img.reshape(-1,1)
    relation = np.matmul(img_flat,img_flat.transpose())
    cv2.imshow("1",relation)
    cv2.waitKey(0)
    None
