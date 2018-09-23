import cv2
import numpy as np
from numpy.linalg import norm
import sys
import os
import json
from matplotlib import pyplot as plt
import copy

#这特么是一堆神奇的魔法数字  
SZ = 20          #训练图片长宽
MAX_WIDTH = 1000 #原始图片最大宽度
Min_Area = 2000  #车牌区域允许最小面积
PROVINCE_START = 1000

#读取图片文件
def imreadex(filename):
	return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
	
def point_limit(point):
	if point[0] < 0:
		point[0] = 0
	if point[1] < 0:
		point[1] = 0

#根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
	up_point = -1#上升点
	is_peak = False
	if histogram[0] > threshold:
		up_point = 0
		is_peak = True
	wave_peaks = []

	for i,x in enumerate(histogram):
		if is_peak and x < threshold:#如果当前状态是波峰，且当前位置的值小于阈值
			if i - up_point > 2:#但前位置与上升点的距离大于2
				is_peak = False#波峰到此位置
				wave_peaks.append((up_point, i))#记录下这个波峰的起始位置和终止位置
		elif not is_peak and x >= threshold:#如果当前状态不是波峰，且当前位置的值大于阈值
			is_peak = True#将当前状态修改为：处于波峰
			up_point = i#记录上升点位置

	#在这里有一个神奇的事情，python中定义于for循环内部的变量，在for循环外部仍然可见

	#记录下最后一个波峰
	if is_peak and up_point != -1 and i - up_point > 4:
		wave_peaks.append((up_point, i))

	return wave_peaks

#根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
	part_cards = []
	for wave in waves:
		part_cards.append(img[:, wave[0]:wave[1]])
	return part_cards

#来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11']/m['mu02']
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

#来自opencv的sample，用于svm训练
#好像是在求梯度直方图
def preprocess_hog(digits):
	samples = []
	for img in digits:
		#对输入中的每一幅图像求梯度直方图
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		
		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		
		samples.append(hist)
	return np.float32(samples)

#不能保证包括所有省份
provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]
class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)  
	def save(self, fn):
		self.model.save(fn)

class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
	#训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
	#字符识别
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()




class CardPredictor:
	def __init__(self):
		#车牌识别的部分参数保存在js中，便于根据图片分辨率做调整
		f = open('config.js')
		j = json.load(f)
		for c in j["config"]:
			if c["open"]:
				self.cfg = c.copy()
				break
		else:
			raise RuntimeError('没有设置有效配置参数')

	def __del__(self):
		self.save_traindata()
	def train_svm(self):
		#识别英文字母和数字
		self.model = SVM(C=1, gamma=0.5)
		#识别中文
		self.modelchinese = SVM(C=1, gamma=0.5)
		if os.path.exists("svm.dat"):
			self.model.load("svm.dat")
		else:
			chars_train = []
			chars_label = []
			
			for root, dirs, files in os.walk("train\\chars2"):
				if len(os.path.basename(root)) > 1:
					continue
				root_int = ord(os.path.basename(root))
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(root_int)
			
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.model.train(chars_train, chars_label)
		if os.path.exists("svmchinese.dat"):
			self.modelchinese.load("svmchinese.dat")
		else:
			chars_train = []
			chars_label = []
			for root, dirs, files in os.walk("train\\charsChinese"):
				if not os.path.basename(root).startswith("zh_"):
					continue
				pinyin = os.path.basename(root)
				index = provinces.index(pinyin) + PROVINCE_START + 1 #1是拼音对应的汉字
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(index)
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)
			#chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.modelchinese.train(chars_train, chars_label)

	def save_traindata(self):
		if not os.path.exists("svm.dat"):
			self.model.save("svm.dat")
		if not os.path.exists("svmchinese.dat"):
			self.modelchinese.save("svmchinese.dat")


	#根据颜色信息精确定位车牌位置
	def accurate_place(self, card_img_hsv, limit1, limit2, color):
		#初始车牌区域图像的行数和列数
		row_num, col_num = card_img_hsv.shape[:2]

		xl = col_num
		xr = 0
		yb = 0
		yt = row_num
		#col_num_limit = self.cfg["col_num_limit"]
		row_num_limit = self.cfg["row_num_limit"]
		col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5#绿色有渐变


		#根据颜色确定车牌图像的上下边界
		for i in range(row_num):
			count = 0
			for j in range(col_num):
				H = card_img_hsv.item(i, j, 0)
				S = card_img_hsv.item(i, j, 1)
				V = card_img_hsv.item(i, j, 2)
				if limit1 < H <= limit2 and 34 < S and 46 < V:
					count += 1

			if count > col_num_limit:
				if yt > i:
					yt = i
				if yb < i:
					yb = i

		#根据颜色确定车牌图像的左右边界
		for j in range(col_num):
			count = 0
			for i in range(row_num):
				H = card_img_hsv.item(i, j, 0)
				S = card_img_hsv.item(i, j, 1)
				V = card_img_hsv.item(i, j, 2)
				if limit1 < H <= limit2 and 34 < S and 46 < V:
					count += 1

			if count > row_num - row_num_limit:
				if xl > j:
					xl = j
				if xr < j:
					xr = j
					
		return xl, xr, yb, yt
		





	#算法的核心部分
	#先使用图像边缘和车牌颜色定位车牌，再用SVM识别字符
	def predict(self, car_pic):

		########################################################边缘轮廓提取（开始）#######################################################
		if type(car_pic) == type(""):
			#如果输入是一个字符串，则从它指向的路径读取图片
			img = imreadex(car_pic)
		else:
			#否则，认为输入就是一张图片
			img = car_pic

		#显示原始图像
		original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
		# plt.figure("原始图像")
		# plt.imshow(original)
		# plt.axis('off')
		

		pic_hight, pic_width = img.shape[:2]

		#如果输入图像过大，则调整图像大小
		if pic_width > MAX_WIDTH:
			resize_rate = MAX_WIDTH / pic_width
			img = cv2.resize(img, (MAX_WIDTH, int(pic_hight*resize_rate)), interpolation=cv2.INTER_AREA)
		
		blur = self.cfg["blur"]

		#高斯模糊去噪
		if blur > 0:
			img = cv2.GaussianBlur(img, (blur, blur), 0)
		oldimg = img
		# oldimg = original
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#equ = cv2.equalizeHist(img)
		#img = np.hstack((img, equ))

		#高斯模糊后的灰度图像
		# plt.figure("高斯模糊后的灰度图像")
		# plt.imshow(img, cmap = 'gray')
		# plt.axis('off')
		

		#去掉图像中不会是车牌的区域
		kernel = np.ones((20, 20), np.uint8)#结构元素
		img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)#形态学开运算

		
		# plt.figure("开运算后的灰度图像")
		# plt.imshow(img_opening, cmap = 'gray')
		# plt.axis('off')
		
		
		img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)#原图与开运算后的图像相减

		# plt.figure("原灰度图减去开运算图")
		# plt.imshow(img_opening, cmap = 'gray')
		# plt.axis('off')
	

		#找到图像边缘
		ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#大津阈值
		img_edge = cv2.Canny(img_thresh, 100, 200)#canny算子提取边缘
		
		# 画出提取的图像边缘
		# plt.figure("初始边缘图")
		# plt.imshow(img_edge, cmap = 'gray')
		# plt.axis('off')
		



		#边缘整体化，使用开运算和闭运算让图像边缘成为一个整体
		kernel = np.ones((self.cfg["morphologyr"], self.cfg["morphologyc"]), np.uint8)
		img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
		img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
		
		#画出整合后的边缘图
		# plt.figure("整合后的边缘图")
		# plt.imshow(img_edge2, cmap = 'gray')
		# plt.axis('off')
		# plt.show()
		
		
		
		#查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
		#coutours里面保存的是轮廓里面的点，cv2.CHAIN_APPROX_SIMPLE表示：只保存角点
		image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#检测物体轮廓
		tmp1 = image
		tmp1 = cv2.cvtColor(tmp1, cv2.COLOR_GRAY2BGR)#灰度图转化为彩色图
		#参数-1表示画出所有轮廓
		#参数(0, 255, 0)表示为轮廓上色的画笔颜色为绿色
		#参数2表示画笔的粗细为2
		cv2.drawContours(tmp1, contours, -1, (0, 255, 0), 2)
		# 画出分割出来的轮廓
		# plt.figure("标记出轮廓")
		# plt.imshow(tmp1)
		# plt.axis('off')




		#挑选出面积大于Min_Area的轮廓
		contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]
		print('len(contours)', len(contours))
		tmp2 = image
		tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_GRAY2BGR)#灰度图转化为彩色图
		cv2.drawContours(tmp2, contours, -1, (0, 255, 0), 2)
		# 画出剔除面积较小的轮廓后的轮廓图
		plt.figure("剔除面积较小的轮廓")
		plt.imshow(tmp2)
		plt.axis('off')
		# plt.show()
		############################################################边缘轮廓提取（结束）#######################################################





		###########################################################计算车牌可能出现矩形区域（开始）###############################################
		car_contours = []
		tmp3 = image
		tmp3 = cv2.cvtColor(tmp3, cv2.COLOR_GRAY2RGB)#灰度图转化为彩色图
		tmp4 = copy.deepcopy(oldimg)
		tmp4 = cv2.cvtColor(tmp4, cv2.COLOR_BGR2RGB)
		for cnt in contours:
			#使用cv2.minAreaRect函数生成每个轮廓的最小外界矩形
			#输入为表示轮廓的点集
			#返回值rect中包含最小外接矩形的中心坐标，宽度高度和旋转角度（但是这里的宽度和高度不是按照其长度来定义的）
			rect = cv2.minAreaRect(cnt)
			area_width, area_height = rect[1]
			#做一定调整，保证宽度大于高度
			if area_width < area_height:
				area_width, area_height = area_height, area_width
			wh_ratio = area_width / area_height
			#print(wh_ratio)

			#要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
			if wh_ratio > 2 and wh_ratio < 5.5:
				car_contours.append(rect)
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				cv2.drawContours(tmp3, [box], -1, (0, 255, 0), 2)
				cv2.drawContours(tmp4, [box], -1, (0, 255, 0), 2)

		print(len(car_contours))
		#在轮廓图中画出选出来的车牌区域大致位置
		# plt.figure("车牌大致定位（轮廓图）")
		# plt.imshow(tmp3)
		# plt.axis('off')


		#在原图中画出选出来的车牌区域大致位置
		plt.figure("车牌大致定位（原图）")
		plt.imshow(tmp4)
		plt.axis('off')
		# plt.show()
		
		#########################################################计算车牌可能出现矩形区域（结束）##############################################
		


		######################################################将倾斜的矩形调整为不倾斜（开始）###################################################
		# print("精确定位")
		print(rect)
		card_imgs = []
		#矩形区域可能是倾斜的矩形，需要矫正，以便使用颜色定位
		for rect in car_contours:

			#调整角度，使得矩形框左高右低

			#0度和1度之间的所有角度当作1度处理,-1度和0度之间的所有角度当作-1度处理
			#这个处理是必要的，如果不做这个处理的话，后面仿射变换可能会得到一张全灰的图片
			#因为如果角度接近于0,那么矩形四个角点中任意两个角点，其某一个坐标非常接近，这种
			#情况下，哪个角点在最上边，哪个角点在最下边，哪个角点在最左边，哪个角点在最右边，就
			#没有了很强的区分度，所以仿射变换控制点的对应关系，很可能出现错配，造成仿射变换失败
			if rect[2] > -1:
				angle = -1
			else:
				angle = rect[2]
			

			#扩大范围，避免车牌边缘被排除
			rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle)
			box = cv2.boxPoints(rect)

			#bottom_point:矩形框4个角中最下面的点
			#right_point:矩形框4个角中最右边的点
			#left_point：矩形框4个角中最左边的点
			#top_point:矩形框4个角中最上面的点
			bottom_point = right_point = [0, 0]
			left_point = top_point = [pic_width, pic_hight]
			for point in box:
				if left_point[0] > point[0]:
					left_point = point
				if top_point[1] > point[1]:
					top_point = point 
				if bottom_point[1] < point[1]:
					bottom_point = point
				if right_point[0] < point[0]:
					right_point = point


			#这里需要注意的是：cv2.boxPoints检测矩形，返回值中角度的范围是[-90, 0]，所以该函数中并不是长度大的作为底，长度
			#小的作为高，而是以从x轴逆时针旋转，最先到达的边为底，另一条边为高
			#这里为了矫正图像所作的仿射变换只能处理小角度，若角度过大，畸变很严重
			#在该程序里，没有对矩形做旋转，然后再仿射变换，而是直接做仿射变换，所以只有当带识别图片中车牌位于水平位置附近时，
			#才能正确识别，而当车牌绕着垂直于车牌的轴有较大转动时，识别就会失败
			if left_point[1] <= right_point[1]:#正角度
				new_right_point = [right_point[0], bottom_point[1]]

				pts2 = np.float32([left_point, bottom_point, new_right_point])
				pts1 = np.float32([left_point, bottom_point, right_point])

				#用3个控制点进行仿射变换
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
				# plt.figure("仿射变换后的图像")
				# plt.imshow(dst)
				# plt.axis('off')

				point_limit(new_right_point)
				point_limit(bottom_point)
				point_limit(left_point)
				card_img = dst[int(left_point[1]):int(bottom_point[1]), int(left_point[0]):int(new_right_point[0])]
				card_imgs.append(card_img)
				# plt.figure("可能的车牌图像")
				# plt.imshow(card_img)
				# plt.axis('off')
				# plt.figure("仿射变换前的车牌图像")
				# plt.imshow(oldimg[int(left_point[1]):int(bottom_point[1]), int(left_point[0]):int(new_right_point[0])])
				# plt.axis('off')

			elif left_point[1] > right_point[1]:#负角度
				new_left_point = [left_point[0], bottom_point[1]]

				pts2 = np.float32([new_left_point, bottom_point, right_point])
				pts1 = np.float32([left_point, bottom_point, right_point])

				#仿射变换
				M = cv2.getAffineTransform(pts1, pts2)
				dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
				# plt.figure("仿射变换后的图像")
				# plt.imshow(dst)
				# plt.axis('off')

				point_limit(right_point)
				point_limit(bottom_point)
				point_limit(new_left_point)
				card_img = dst[int(right_point[1]):int(bottom_point[1]), int(new_left_point[0]):int(right_point[0])]
				card_imgs.append(card_img)
				# plt.figure("可能的车牌图像")
				# plt.imshow(card_img)
				# plt.axis('off')
				# plt.figure("仿射变换前的车牌图像")
				# plt.imshow(oldimg[int(right_point[1]):int(bottom_point[1]), int(new_left_point[0]):int(right_point[0])])
				# plt.axis('off')
			# plt.show()
		######################################################将倾斜的矩形调整为不倾斜（结束）########################################
		
		




		############################################################判定矩形区域的颜色，目前只识别蓝、绿、黄车牌（开始）######################################
		colors = []
		#enumerate是python的内置函数，对于一个可遍历的对象，enumerate将其组成一个索引序列，利用它可以同时获得索引和值
		for card_index,card_img in enumerate(card_imgs):
			green = yello = blue = black = white = 0
			card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
			#有转换失败的可能，原因来自于上面矫正矩形出错
			if card_img_hsv is None:
				continue
			row_num, col_num= card_img_hsv.shape[:2]
			card_img_count = row_num * col_num

			for i in range(row_num):
				for j in range(col_num):
					H = card_img_hsv.item(i, j, 0)
					S = card_img_hsv.item(i, j, 1)
					V = card_img_hsv.item(i, j, 2)
					if 11 < H <= 34 and S > 34:
						#黄色像素个数
						yello += 1

					elif 35 < H <= 99 and S > 34:
						#绿色像素个数
						green += 1

					elif 100 < H <= 124 and S > 34:
						#蓝色像素个数
						blue += 1
					
					if 0 < H <180 and 0 < S < 255 and 0 < V < 46:
						black += 1
					elif 0 < H <180 and 0 < S < 43 and 221 < V < 225:
						white += 1
			color = "no"

			limit1 = limit2 = 0
			if yello*2 >= card_img_count:
				#黄色像素占像素总数一半以上
				color = "yello"
				limit1 = 11
				limit2 = 34#有的图片有色偏偏绿

			elif green*2 >= card_img_count:
				#绿色像素占像素总数一半以上
				color = "green"
				limit1 = 35
				limit2 = 99

			elif blue*2 >= card_img_count:
				#蓝色像素占像素总数一半以上
				color = "blue"
				limit1 = 100
				limit2 = 124#有的图片有色偏偏紫

			elif black + white >= card_img_count*0.7:#TODO
				#这特么是什么东西
				#莫非是黑白图
				color = "bw"
			colors.append(color)

			# card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
			# plt.figure("车牌图像")
			# plt.imshow(card_img)
			# plt.axis('off')
			# plt.show()

			if limit1 == 0:
				#出现这种情况，说明该矩形区域不是车牌
				#所以，不用进行进一步处理，直接跳过，处理下一个矩形区域
				continue
			#########################################################判定矩形区域的颜色，目前只识别蓝、绿、黄车牌（结束）#############################





			########################################################根据车牌颜色再定位，缩小边缘非车牌边界（开始）#####################################
			xl, xr, yb, yt = self.accurate_place(card_img_hsv, limit1, limit2, color)
			if yt == yb and xl == xr:
				continue
			need_accurate = False
			if yt >= yb:
				yt = 0
				yb = row_num
				need_accurate = True
			if xl >= xr:
				xl = 0
				xr = col_num
				need_accurate = True
			card_imgs[card_index] = card_img[yt:yb, xl:xr] if color != "green" or yt < (yb-yt)//4 else card_img[yt-(yb-yt)//4:yb, xl:xr]
			if need_accurate:#可能x或y方向未缩小，需要再试一次
				card_img = card_imgs[card_index]
				card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
				xl, xr, yb, yt = self.accurate_place(card_img_hsv, limit1, limit2, color)
				if yt == yb and xl == xr:
					continue
				if yt >= yb:
					yt = 0
					yb = row_num
				if xl >= xr:
					xl = 0
					xr = col_num
			card_imgs[card_index] = card_img[yt:yb, xl:xr] if color != "green" or yt < (yb-yt)//4 else card_img[yt-(yb-yt)//4:yb, xl:xr]
			# plt.figure("缩小边界后的车牌图像")
			# plt.imshow(card_imgs[card_index])
			# plt.axis('off')
			# plt.show()
		 	################################################根据车牌颜色再定位，缩小边缘非车牌边界（结束）#####################################
		
		
		




		#######################################################识别车牌中的字符（开始）########################################################
		predict_result = []
		roi = None

		card_color = None
		for i, color in enumerate(colors):
			if color in ("blue", "yello", "green"): 

				#####################################################寻找直方图中的波峰（开始）##########################################
				card_img = card_imgs[i]

				#将三通道的彩色车牌图像转换为灰度图像
				gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

				#黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
				if color == "green" or color == "yello":
					gray_img = cv2.bitwise_not(gray_img)

				#车牌图像二值化
				ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
				
				plt.figure("二值化后的车牌图像")
				plt.imshow(gray_img, cmap = 'gray')
				plt.axis('off')
				# plt.show()
				#查找竖直直方图波峰
				#参数axis为1表示压缩列，将每一行的元素相加，将矩阵压缩为一列
				x_histogram  = np.sum(gray_img, axis=1)
				plt.figure('竖直方向波峰图')
				plt.plot(x_histogram)
				# plt.show()
				x_min = np.min(x_histogram)
				x_average = np.sum(x_histogram)/x_histogram.shape[0]
				x_threshold = (x_min + x_average)/2

				#使用自定义的函数寻找波峰，从而分割字符
				wave_peaks = find_waves(x_threshold, x_histogram)
				if len(wave_peaks) == 0:
					print("peak less 0:")
					continue

				#认为竖直方向，最大的波峰为车牌区域
				wave = max(wave_peaks, key=lambda x:x[1]-x[0])
				gray_img = gray_img[wave[0]:wave[1]]
				plt.figure('字符所在的区域的车牌图像')
				plt.imshow(gray_img, cmap = 'gray')
				
				
				#查找水平直方图波峰
				row_num, col_num= gray_img.shape[:2]
				#去掉车牌上下边缘1个像素，避免白边影响阈值判断
				gray_img = gray_img[1:row_num-1]
				#参数axis为0表示压缩列，将每一列的元素相加，将矩阵压缩为一行
				y_histogram = np.sum(gray_img, axis=0)

				plt.figure('水平方向波峰图')
				plt.plot(y_histogram)
				# plt.show()

				y_min = np.min(y_histogram)
				y_average = np.sum(y_histogram)/y_histogram.shape[0]
				y_threshold = (y_min + y_average)/5#U和0要求阈值偏小，否则U和0会被分成两半

				wave_peaks = find_waves(y_threshold, y_histogram)

				#for wave in wave_peaks:
				#	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2) 
				#车牌字符数应大于6
				if len(wave_peaks) <= 6:
					print("peak less 1:", len(wave_peaks))
					continue
				
				#找出宽度最大的波峰
				wave = max(wave_peaks, key=lambda x:x[1]-x[0])
				max_wave_dis = wave[1] - wave[0]
				
				#判断是否是左侧车牌边缘
				if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis/3 and wave_peaks[0][0] == 0:
					#如果是左侧车牌边缘，则将其剔除
					wave_peaks.pop(0)
				
				#####################################################寻找直方图中的波峰（结束）##########################################



				########################################组合汉字，去除车牌上的分割点（开始）#############################################
				#组合汉字（一个汉字可能由好几个连续波峰组成）
				#一个汉字可能由好几个波峰组成，找到这几个波峰，并且将它们合并在一起
				cur_dis = 0
				for i, wave in enumerate(wave_peaks):
					if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
						break
					else:
						cur_dis += wave[1] - wave[0]
				if i > 0:
					#这种情况说明，前几个波峰的组合代表一个汉字
					wave = (wave_peaks[0][0], wave_peaks[i][1])
					wave_peaks = wave_peaks[i+1:]
					wave_peaks.insert(0, wave)
				
				#去除车牌上的分隔点
				point = wave_peaks[2]
				if point[1] - point[0] < max_wave_dis/3:
					point_img = gray_img[:,point[0]:point[1]]
					if np.mean(point_img) < 255/5:
						wave_peaks.pop(2)
				
				if len(wave_peaks) <= 6:
					print("peak less 2:", len(wave_peaks))
					continue

				#调用自定义的函数，分割车牌中的字符
				part_cards = seperate_card(gray_img, wave_peaks)
				########################################组合汉字，去除车牌上的分割点（结束）############################################




				###########################################识别分割出来的字符（开始）###################################################
				for i, part_card in enumerate(part_cards):
					# print(part_card)
					#可能是固定车牌的铆钉
					if np.mean(part_card) < 255/5:
						print("a point")
						continue
					part_card_old = part_card
					w = abs(part_card.shape[1] - SZ)//2#//运算符取除法结果的最小整数
					
					

					#给图像扩充边界cv2.copyMakeBorder(src,top, bottom, left, right ,borderType,value)
					part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value = [0,0,0])#左右边界用黑色背景扩充
					part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)#将分割出来的图像调整为设定的训练图像的尺寸
							
					ret, part_card = cv2.threshold(part_card, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

					title = "分割出来的字符" + str(i + 1)
					plt.figure(title)
					plt.imshow(part_card, cmap = 'gray')
					plt.axis('off')
					# plt.show()

					#part_card = deskew(part_card)
					#根据分割出的字符图像，计算HOG特征
					part_card = preprocess_hog([part_card])

					if i == 0:
						#如果是第一个字符，则识别汉字
						resp = self.modelchinese.predict(part_card)
						charactor = provinces[int(resp[0]) - PROVINCE_START]
					else:
						#如果不是第一个字符，则识别字母和数字
						resp = self.model.predict(part_card)
						charactor = chr(resp[0])

					#判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
					if charactor == "1" and i == len(part_cards)-1:
						if part_card_old.shape[0]/part_card_old.shape[1] >= 7:#1太细，认为是边缘
							continue

					predict_result.append(charactor)
				roi = card_img
				roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) 
				card_color = color
				break
				###########################################识别分割出来的字符（结束）###################################################

		#######################################################识别车牌中的字符（结束）########################################################


		return predict_result, roi, card_color#识别到的字符、定位的车牌图像、车牌颜色

if __name__ == '__main__':
	c = CardPredictor()
	c.train_svm()
	# r, roi, color = c.predict("test/lLD9016.jpg")
	r, roi, color = c.predict("test/鲁L D9016.jpg")
	# r, roi, color = c.predict("myData/8.jpg")
	# r, roi, color = c.predict("test/皖A 85890.jpg")
	# r, roi, color = c.predict("myData/2.jpg")
	print(r)
	
	