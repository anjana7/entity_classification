from PIL import Image
import cv2
from pytesseract import image_to_string

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def image_clear():

    im = Image.open("/home/anjana/Anjana/testdata/HelloLeads - BCR/Biz cards Set 2 Preethi 29 Jun 2017/Img_6621.jpeg")
    
    im.save("/home/anjana/Anjana/testdata/results/res_Img_77.jpg", dpi=(600,600))
    #im.show()
    
    im_contr = change_contrast(im, 100)
    #im_contr.show()
    
    gray_img = cv2.cvtColor(np.asarray(im_contr), cv2.COLOR_RGB2GRAY)
    kernel = np.ones((1,1), np.uint8)
    img = cv2.dilate(gray_img, kernel, iterations = 1)
    img = cv2.erode(img, kernel, iterations = 1)
    
    cv2.imwrite('/home/anjana/Anjana/testdata/Mano_test1.jpg', img)
    
    basewidth = 870
    img = Image.open('/home/anjana/Anjana/testdata/Mano_test1.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save('/home/anjana/Anjana/testdata/Mano_test1.jpg') 
    

    img = cv2.adaptiveThreshold(np.asarray(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
    #imshow(img)
    
    img = np.invert(img) 
    cv2.imwrite('/home/anjana/Anjana/testdata/Mano_test1.jpg', img)
    result = image_to_string(Image.open('/home/anjana/Anjana/testdata/Mano_test1.jpg'))
    
    return(result)        


image_clear()
