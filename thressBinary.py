import cv2
import numpy as np
import glob
#path_to_img = './neg/193.jpg'
#path = './neg'
path = './green'
def segment_by_color():

	global path
	counter = 0
	for pic in glob.glob("{}/*.jpg".format(path)):


		img  = cv2.imread(pic,0)
		img_g  = cv2.resize(img, (8*3,24*3), interpolation = cv2.INTER_CUBIC)


		filter_g = cv2.bilateralFilter(img_g,35,15,15)
		filter_g = cv2.GaussianBlur(filter_g,(15,15),0)


		#im_gray = cv2.imread('gray_image.png', cv2.IMREAD_GRAYSCALE)
		thresh = 127
		img_bw = cv2.threshold(filter_g, thresh, 255, cv2.THRESH_BINARY)[1]


		

###
		

###
		#img_bw =  np.stack((img_bw,)*3, -1)

		print('MEDIAN VALUE IS', np.mean(img_bw))
		if np.mean(img_bw) < 8 : 
			cv2.imwrite('./out_else/else_{}.jpg'.format(counter), img_bw)
		else:
			cv2.imwrite('./out/green_{}.jpg'.format(counter), img_bw)
		counter +=1
		"""
		img =  np.stack((img,)*3, -1)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		# GREEN
		lower_green = np.array([40,20,0], dtype=np.uint8)		# OPEN Green channels
		upper_green = np.array([90,255,255 ], dtype=np.uint8)   #95.255.255  ideal my square.


		#MASKS
		mask_green = cv2.inRange(hsv, lower_green, upper_green)
		
		full_mask = mask_green

		res_g = cv2.bitwise_and(img, img, mask = full_mask)
		res_g = cv2.bilateralFilter(res_g, 35,75,75)
		res_g = cv2.GaussianBlur(res_g,(15,15),0)

		cv2.imwrite('./out/green_{}.jpg'.format(counter), res_g)
		counter = counter + 1
		print('working in...',counter)
		"""
	print('done!')




def thres_one_image(pic):
    img  = cv2.imread(pic)
    img = cv2.resize(img, (100,200))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # YELLOW /(Orangen)

    lower_yellow = np.array([18,40,190], dtype=np.uint8)
    upper_yellow =np.array([27,255,255], dtype=np.uint8)


    # RED

    lower_red = np.array([140,100,0], dtype=np.uint8)
    upper_red = np.array([180,255,255], dtype=np.uint8)

    # GREEN
    lower_green = verde1=np.array([70,150,0], dtype=np.uint8)
    upper_green = np.array([90,255,255], dtype=np.uint8)

    #MASKS

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    full_mask = mask_red + mask_yellow + mask_green

    res = cv2.bitwise_and(img,img, mask= full_mask)

    cv2.imwrite('out_3.jpg',res)

#pic = './3.png'
#thres_one_image(pic)

segment_by_color()