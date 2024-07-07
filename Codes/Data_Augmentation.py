import cv2

def Data_Augmentation():

    images_path = "/content/drive/MyDrive/UNet-Implementation-main/UNet-Implementation-main/data/train_images"
    masks_path = "/content/drive/MyDrive/UNet-Implementation-main/UNet-Implementation-main/data/train_masks"

    L = 101 #Number of original images


    for i in range(1,L+1):
    image_path = images_path + "/" + str(i) + ".jpg"
    mask_path = masks_path + "/" + str(i) + ".jpg"

    img = cv2.imread(image_path)

    mask = cv2.imread(mask_path)

    #Horizontal flipping
    img_hflip = cv2.flip(img,1)
    mask_hflip = cv2.flip(mask,1)

    cv2.imwrite(images_path  +  "/" + str(i+L) + ".jpg", img_hflip)
    cv2.imwrite(masks_path  +  "/" +str(i+L) + ".jpg", mask_hflip)

    #Vertical flipping
    img_vflip = cv2.flip(img,0)
    mask_vflip = cv2.flip(mask,0)

    cv2.imwrite(images_path  +  "/" +str(i+L*2) + ".jpg", img_vflip)
    cv2.imwrite(masks_path +  "/" +str(i+L*2) + ".jpg", mask_vflip)

    #90 rotation
    img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    mask_90 = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite(images_path +  "/" +str(i+L*3) + ".jpg", img_90)
    cv2.imwrite(masks_path +  "/" + str(i+L*3) + ".jpg", mask_90)

    #-90 rotation
    img_m90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    mask_m90 = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    cv2.imwrite(images_path  +  "/" +str(i+L*4) + ".jpg", img_m90)
    cv2.imwrite(masks_path +  "/" +str(i+L*4) + ".jpg", mask_m90)

    # add gaussian noise to img
    img_noise = cv2.GaussianBlur(img,(5,5),0)

    cv2.imwrite(images_path  + "/" + str(i+L*5) + ".jpg", img_noise)
    cv2.imwrite(masks_path +  "/" +str(i+L*5) + ".jpg", mask)
