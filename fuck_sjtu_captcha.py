#coding:utf-8

"""
验证码处理步骤:

1. 二值化
2. 去噪点(由于sjtu验证码没有噪点，不需要这步)
3. 字符切割
4. 单个字符图片旋转到合适角度:旋转卡壳算法（投影至x轴长度最小）(效果不好，sjtu的验证码都没什么旋转，暂时不用后续再加)
5. 缩放到相同大小
6. 持久化，string hickle
"""
from PIL import Image
import numpy as np
import os


from utils import (
	COLOR_RGB_BLACK, COLOR_RGB_WHITE, COLOR_RGBA_BLACK, COLOR_RGBA_WHITE,
	BORDER_LEFT, BORDER_TOP, BORDER_RIGHT, BORDER_BOTTOM,
	RAW_DATA_DIR, PROCESSED_DATA_DIR,
	NORM_SIZE,
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
		child_images = self._cut_images()
		for i in range(len(child_images)):
			# self._resize_to_norm(child_images[i]).show()
			self._captcha_to_string(self._resize_to_norm(child_images[i]), save_as = '%d' % i)

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

		# 切割图片
		croped_images = []
		height = self._image.size[1]

		for start_x, width in split_seq:
			# 同时去掉y轴上下多余的空白
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
	
	def _get_black_border(self, image):
		"""
		获取指定图像的内容边界坐标
		:param image: 图像 Image Object
		:return: 图像内容边界坐标tuple (left, top, right, bottom)
		"""
		width, height = image.size
		max_x = max_y = 0
		min_x = width - 1
		min_y = height - 1
		for y in range(height):
			for x in range(width):
				if image.getpixel((x, y)) == COLOR_RGBA_BLACK:
					min_x = min(min_x, x)
					max_x = max(max_x, x)
					min_y = min(min_y, y)
					max_y = max(max_y, y)
		return min_x, min_y, max_x, max_y

	def _rotate_image(self, image):
		"""
		将单个字符图片旋转到合适角度 (投影至X轴长度最小)
		:return: 旋转后的图像 (RGB)
		"""
		image = image.convert('RGBA')
		optimisim_image = image
		for angle in range(-30, 31):
			image_copy = image.rotate(angle, expand=True)
			fff = Image.new('RGBA', image_copy.size, (255, )*4)
			out = Image.composite(image_copy, fff, image_copy)

			border_out = self._get_black_border(out)
			border_optimisim = self._get_black_border(optimisim_image)
			if border_out[BORDER_RIGHT] - border_out[BORDER_LEFT] + 1 < border_optimisim[BORDER_RIGHT] - border_optimisim[BORDER_LEFT] + 1:
				optimisim_image = out

		border = self._get_black_border(optimisim_image)
		optimisim_image = optimisim_image.crop((
		    border[BORDER_LEFT],
		    border[BORDER_TOP],
		    border[BORDER_RIGHT],
		    border[BORDER_BOTTOM]
		))
		optimisim_image = optimisim_image.convert('RGB')
		return optimisim_image

	def _resize_to_norm(self, image):
		"""
		将单个图像缩放至32x32像素标准图像
		:param image: 图像 (RGB)
		:return: 缩放后的Image Object
		"""
		if image.size[0] > NORM_SIZE or image.size[1] > NORM_SIZE:
			image = image.resize((NORM_SIZE, NORM_SIZE))
		width, height = image.size
		new_image = Image.new('RGB', (NORM_SIZE, NORM_SIZE), COLOR_RGB_WHITE)
		offset = ((NORM_SIZE - width) / 2, (NORM_SIZE - height) / 2)
		new_image.paste(image, offset)
		return new_image

	def _captcha_to_2d_list(self, image):
		"""
		将验证码转换为数字编码
		:param image: 图像
		:return: 数字编码字符串
		"""
		if image.size != (NORM_SIZE, NORM_SIZE):
			raise Exception("Image needs to normalize before to string")
        
		# 将pixel写到二维数组中
		data = []
		for x in range(0, NORM_SIZE):
			data.append([])
			for y in range(0, NORM_SIZE):
				data[-1].append(0)

		for y in range(0, NORM_SIZE):
			for x in range(0, NORM_SIZE):
				data[y][x] = 1 if image.getpixel((x, y)) == COLOR_RGB_BLACK else 0

		return data

	def _captcha_to_string(self, image, save_as):
		data = self._captcha_to_2d_list(image)
		# 写到文件: data的数据类型必须是str(map转换)
		with open(save_as, 'w') as outfile:
			for row in data:
				outfile.write(''.join(map(str, row)) + '\n')

	def _captcha_to_array(self, image):
		data = self._captcha_to_2d_list(image)
		return np.asarray(data).reshape(1, -1).flatten()


def main():
	train_data = np.zeros(shape = (1500, NORM_SIZE*NORM_SIZE))
	# for i in xrange(1500):
	myCaptcha = SJTUCaptcha(os.path.join(RAW_DATA_DIR, '%d.jpg'%10))
	myCaptcha.preprocess()
	
if __name__ == '__main__':
	main()
