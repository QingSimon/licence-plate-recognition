import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import predict
import cv2
from PIL import Image, ImageTk
import time



class Surface(ttk.Frame):
	pic_path = ""
	viewHeight = 600
	viewWidth = 600
	update_time = 0
	color_transform = {"green":("绿牌","#55FF55"), "yello":("黄牌","#FFFF00"), "blue":("蓝牌","#6666FF")}

	#构建图形界面	
	def __init__(self, win):
		ttk.Frame.__init__(self, win)
		#界面名称
		win.title("车牌识别")
		win.state("normal")
		self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")

		#创建3块矩形区域，作为容器来布局图形界面
		############################################frame_left主要用来显示待识别的图像##################################################   
		frame_left = ttk.Frame(self)
		frame_left.pack(side = LEFT, expand = 1, fill = BOTH)

		ttk.Label(frame_left, text = 'input picture：', font = ('Arial', 20)).pack(anchor = "nw", padx = 30)
		self.image_ctl = ttk.Label(frame_left)
		self.image_ctl.pack(anchor = "nw")
		############################################################################################################################



		############################################frame_right1主要用来显示识别结果##################################################
		frame_right1 = ttk.Frame(self)
		frame_right1.pack(side=TOP,expand=1,fill=tk.Y)

		ttk.Label(frame_right1, text='license plate segmentation：', font = ('Arial', 20)).grid(column = 0, row = 0, sticky = tk.W)
		
		self.roi_ctl = ttk.Label(frame_right1, font = ('Arial', 20))
		self.roi_ctl.grid(column = 0, row = 1)

		ttk.Label(frame_right1, text='recognition result：', font = ('Arial', 20)).grid(column = 0, row = 10, sticky = tk.SW)
		
		self.r_ctl = ttk.Label(frame_right1, text = "", font = ('Arial', 20))
		self.r_ctl.grid(column = 0, row = 11)

		# self.color_ctl = ttk.Label(frame_right1, text = "", font = ('Arial', 20), width = "20")
		# self.color_ctl.grid(column = 0, row = 12, sticky = tk.SW)
		############################################################################################################################





		############################################frame_right2主要用来选择倒入图片的方式##################################################
		frame_right2 = ttk.Frame(self)
		frame_right2.pack(side = RIGHT, expand = 0)

		ttk.Style().configure('TButton', font = ('Arial', 20))
		#创建一个button，用于从文件路径中读取待识别图片，回调函数为from_folder
		from_folder_ctl = ttk.Button(frame_right2, text="chose picture from folder", width = 30, command = self.from_folder)
		from_folder_ctl.pack(anchor = "se", pady = "5")
		############################################################################################################################
		
		
		self.predictor = predict.CardPredictor()
		self.predictor.train_svm()


	#对读入的图像,调整图像大小	
	def get_imgtk(self, img_bgr):
		img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		im = Image.fromarray(img)
		imgtk = ImageTk.PhotoImage(image = im)
		width = imgtk.width()
		height = imgtk.height()
		if width > self.viewWidth or height > self.viewHeight:
			wide_factor = self.viewWidth / width
			high_factor = self.viewHeight / height
			factor = min(wide_factor, high_factor)
			width = int(width * factor)
			if width <= 0 : width = 1
			height = int(height * factor)
			if height <= 0 : height = 1
			im = im.resize((width, height), Image.ANTIALIAS)
			imgtk = ImageTk.PhotoImage(image = im)
		return imgtk


	#在界面上显示分割出来的车牌
	def show_roi(self, r, roi, color):
		if r :
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
			roi = Image.fromarray(roi)
			self.imgtk_roi = ImageTk.PhotoImage(image = roi)
			self.roi_ctl.configure(image = self.imgtk_roi, state = 'enable')

			r = ' '.join(r[:2]) + '   ' + ' '.join(r[2:])
			#显示识别出来的车牌内容
			self.r_ctl.configure(text = r)

			#应该是更新时间记录为，当前系统时间
			self.update_time = time.time()

			#显示识别出来的车牌颜色
			# try:
			# 	c = self.color_transform[color]
			# 	self.color_ctl.configure(text = str(c[0]), background = c[1], state = 'enable')			
			# except: 
			# 	self.color_ctl.configure(state = 'disabled')

		#elif self.update_time + 8 < time.time():
		else:
			#self.roi_ctl.configure(state='disabled')
			self.roi_ctl.configure(image = "", text = "failed!")
			self.r_ctl.configure(text = "failed!")
			#self.color_ctl.configure(state='disabled')
			#self.color_ctl.configure(text = "")

	
		
	def from_folder(self):
		self.pic_path = askopenfilename(title="选择待识别图片", filetypes=[("jpg图片", "*.jpg")])

		#如果图片打开成功
		if self.pic_path:
			img_bgr = predict.imreadex(self.pic_path)
			self.imgtk = self.get_imgtk(img_bgr)
			
			#显示图片
			self.image_ctl.configure(image = self.imgtk)

			#对图片进行处理，得到
			r, roi, color = self.predictor.predict(img_bgr)

			#显示分割出来的车牌
			self.show_roi(r, roi, color)

	
		

#销毁窗口时的操作		
def close_window():
	print("destroy")
	win.destroy()
	
	
if __name__ == '__main__':
	win = tk.Tk()
	surface = Surface(win)
	win.protocol('WM_DELETE_WINDOW', close_window)
	win.mainloop()
	
