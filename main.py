import cv2
import numpy as np
import matplotlib.pyplot as plt


def smoothing(img):
    '''
    perform image smoothing using gaussian filter

    inputs :
    img (ndarray): input Grayscale image

    outputs :
    output (ndarray) : smoothed image
    '''
    output = img.copy()
    output = cv2.GaussianBlur(img, (7, 7), 0)
    return output


def comput_gradient(img):
    '''
    compute image gradiend magnitude and angle using sobel filter

    inputs :
    img (ndarray): input Grayscale image

    outputs :
    mag (ndarray) : gradient magnitude
    angle (ndarray) : gradient angle
    '''

    mag = np.zeros_like(img)
    angle = np.zeros_like(img)

    gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0)
    gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1)

    mag[:, :] = np.sqrt(gx**2 + gy**2)
    angle[:, :] = np.arctan2(gy, gx)

    return mag, angle


def NMS(mag, angle):
    '''
    perform non-maximum suppression

    inputs :
    mag (ndarray) : gradient magnitude
    angle (ndarray) : gradient angle

    outputs :
    output (ndarray) : one-pixel width edges
    '''
    output = mag.copy()

    map_matrix = np.zeros((mag.shape[0], mag.shape[1], 2))
    map_matrix[(angle >= 0) & (angle < 22.5)] = (0, 1)
    map_matrix[(angle >= 22.5) & (angle < 67.5)] = (-1, 1)
    map_matrix[(angle >= 67.5) & (angle < 112.5)] = (-1, 0)
    map_matrix[(angle >= 112.5) & (angle < 157.5)] = (-1, -1)
    map_matrix[(angle >= 157.5) & (angle < 202.5)] = (0, 1)
    map_matrix[(angle >= 202.5) & (angle < 247.5)] = (-1, 1)
    map_matrix[(angle >= 247.5) & (angle < 292.5)] = (-1, 0)
    map_matrix[(angle >= 292.5) & (angle < 337.5)] = (-1, -1)
    map_matrix[(angle >= 337.5) & (angle < 360)] = (0, 1)

    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if output[i][j] == 0:
                continue

            m, n = ((i, j) + map_matrix[i][j]).astype(int)
            if not (m < 0 or m > mag.shape[0]-1 or n < 0 or n > mag.shape[1]-1):
                if output[i][j] < output[m][n]:
                    output[i][j] = 0
                    continue

            m, n = ((i, j) + map_matrix[i][j] * -1).astype(int)
            if not (m < 0 or m > mag.shape[0]-1 or n < 0 or n > mag.shape[1]-1):
                if output[i][j] < output[m][n]:
                    output[i][j] = 0
                    continue

    return output


def hysteresis_threshold(edges, min_th, max_th):
    '''
    perform two-steps threshold

    inputs :
    edges (ndarray) : edges of image
    min_th (int) : weak threshold
    max_th (int) : strong threshold

    outputs :
    output (ndarray) : final edge image
    '''
    output = edges.copy()
    output[ output < min_th] = 0
    output[(output >= min_th) & (output < max_th)] = 0.5
    output[output >= max_th] = 1

    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if output[i][j] == 0 or output[i][j] == 0.5:
                continue

            buffer = list()
            buffer.append((i, j))

            while len(buffer) > 0:
                for b in buffer:
                    output[b[0]][b[1]] = 1
                    buffer.remove(b)
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            w = b[0] + m
                            q = b[1] + n
                            if not (w < 0 or w > edges.shape[0]-1 or q < 0 or q > edges.shape[1]-1):
                                if output[w][q] == 0.5:
                                    buffer.append((w, q))

    output[output < 1] = 0
    output[output == 1] = 255

    return output


def canny(img, min_th=40, max_th=200):
    """
    Perform Canny edge detector
    
    inputs:
    img (ndarray): input Grayscale image
    min_th (int) : weak threshold
    max_th (int) : strong threshold
    
    outputs:
    final_edges (ndarray): edges
    """
    smooth_img = smoothing(img)
    mag, angle = comput_gradient(smooth_img)
    edges = NMS(mag, angle)
    final_edges = hysteresis_threshold(edges, min_th, max_th)
    return final_edges


if __name__ == "__main__":
    img = cv2.imread('image.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    min_th, maxt_th = 40, 125
    edges = canny(image_g, min_th, maxt_th)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('input image')
    plt.figure(figsize=(5, 5))
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.title('Canny')
    plt.show()
