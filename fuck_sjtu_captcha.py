#coding:utf-8

"""
验证码处理步骤:

1. 二值化
2. 去噪点(由于sjtu验证码没有噪点，不需要这步)
3. 字符切割
"""
from PIL import Image
import os
from utils import (
	COLOR_RGB_BLACK, COLOR_RGB_WHITE, COLOR_RGBA_BLACK, COLOR_RGBA_WHITE,
	BORDER_LEFT, BORDER_TOP, BORDER_RIGHT, BORDER_BOTTOM,
	RAW_DATA_DIR, PROCESSED_DATA_DIR,
)

# 存放处理后的图片数据
if not os.path.exists(PROCESSED_DATA_DIR):
	os.mkdir(PROCESSED_DATA_DIR)

class SJTUCaptcha(object):
	def __init__(self, image):
		"""
		初始化
		:param image: 验证码图片文件 Image Object
		:param manual: 是否人工验证, 默认为False, 采用机器验证
		"""
		if isinstance(image, file) or isinstance(image, str) or isinstance(image, unicode):
			self._image = Image.open(image)
		elif isinstance(image, JpegImageFile):
			self._image = image
		else:
			raise Exception('captcha image file is unavailable')

	def preprocess(self):
		# 获取验证码预处理结果
		self._binaryzation()
		for i in self._cut_images():
			i.show()
		

	def _binaryzation(self):
		"""
		将图片进行二值化
		"""
		#有很多种算法，这里选择rgb加权平均值算法
		width, height = self._image.size
		for y in xrange(height):
			for x in xrange(width):
				r, g, b = self._image.getpixel((x, y))
				value = 0.299 * r + 0.587 * g + 0.114 * b
				#value就是灰度值，这里使用127作为阀值，
				#小于127的就认为是黑色也就是0 大于等于127的就是白色，也就是255
				if value < 127:
					self._image.putpixel((x, y), COLOR_RGB_BLACK)
				else:
					self._image.putpixel((x, y), COLOR_RGB_WHITE)
	
	# 图片到x轴或y轴的投影，如果有数据（黑色像素点）值为1，否则为0
	def _get_projection_x(self): # axis = 0: x轴, axis = 1: y轴
		# 初始化投影标记list
		p_x = [0 for _ in xrange(self._image.size[0])]
		width, height = self._image.size
		
		for x in xrange(width):
			for y in xrange(height):
				if self._image.getpixel((x, y)) == COLOR_RGB_BLACK:
					p_x[x] = 1
					break
		return p_x


	# 获取切割后的x轴坐标点，返回值为[初始位置，长度]的列表
	# crop((start_x, start_y, start_x + width, start_y + height))
	def _get_split_seq(self, projection_x):
		split_seq = []
		start_x = 0
		length = 0
		for pos_x, val in enumerate(projection_x):
			if val == 0 and length == 0:
				continue
			elif val == 0 and length != 0:
				split_seq.append([start_x, length])
				length = 0
			elif val == 1:
				if length == 0:
					start_x = pos_x
				length += 1
			else:
				raise Exception('generating split sequence occurs error')
		return split_seq

	def _cut_images(self):
		"""
		切割图像为单个字符块
		:return: list对象, 每个元素为一个单独字符的Image Object
		"""
		# _image.size返回的是(width, height)
		split_seq = self._get_split_seq(self._get_projection_x())
		length = len(split_seq)

		

		# 切割图片
		croped_images = []

		height = self._image.size[1]
		for start_x, width in split_seq:
			begin_row = 0
			end_row = height - 1
			for row in range(height):
				flag = True
				for col in range(start_x, start_x + width):
					if self._image.getpixel((col, row)) == COLOR_RGB_BLACK:
						flag = False
						break
				if not flag: # 如果在当前行找到了黑色像素点，就是起始行
				    begin_row = row
				    break
			for row in reversed(range(height)):
				flag = True
				for col in range(start_x, start_x + width):
					if self._image.getpixel((col, row)) == COLOR_RGB_BLACK:
						flag = False
						break
				if not flag:
					end_row = row
					break
			croped_images.append(self._image.crop((start_x, begin_row, start_x + width, end_row + 1)))
		
		return croped_images

def main():
	myCaptcha = SJTUCaptcha(os.path.join(RAW_DATA_DIR, '%d.jpg'%15))
	myCaptcha.preprocess()
	

if __name__ == '__main__':
	main()
