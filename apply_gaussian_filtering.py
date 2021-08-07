import numpy as np
import os
import cv2


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
    #         print(img.shape)
        return img


def circle_crop(img, sigmaX):
    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width/2)
    y = int(height/2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(
        img, (0, 0), sigmaX), -4, 128)
    return img


def preprocess_image():
    for dirname, _, filenames in os.walk('reference_images\original'):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            img = cv2.imread(os.path.join(dirname, filename))
            img = circle_crop(img, 5)
            cv2.imwrite('Diabetic Retinopathy Detection/gaussian_filtered_images/' + str(filename) +
                        '.jpg', cv2.resize(img, (224, 224)))


if __name__ == '__main__':
    preprocess_image()
