import os
from os import listdir
from shutil import copyfile
import shutil
from PIL import Image
import skimage

in_dir = "/m2docs/mtg-gan/imgs_sorted"
out_dir = "/m2docs/mtg-gan/data"
test_quantity = 0
size = (550,460)
crop = True
i = 0

for color in ['W','U','B','R','G']:
	for tp in ['Creature','Enchantment','Instant','Sorcery','Land']:
		counter = 0
		im_list = os.listdir(os.path.join(in_dir+'/'+color+'/'+tp+'/'))

		label = tp+'_'+color

		if not os.path.exists(out_dir + '/train/' + label):
			os.makedirs(out_dir + '/train/' + label)
		if not os.path.exists(out_dir + '/test/' + label):
			os.makedirs(out_dir + '/test/' + label)
		
		for image in im_list:
			pic = Image.open(os.path.join(in_dir+'/'+color+'/'+tp+'/',image))
			if crop:
				out = pic.crop((0,0,size[0],size[1]))
			else:
				out = pic

			# save to dirs	
			# if counter < test_quantity:
			# 	out.save(os.path.join(out_dir+'/test/'+label,image))
			# else:
				
			out.save(os.path.join(out_dir+'/train/'+label,image))

			print("\rMoving and cropping images [{}/18000ish] [{}]".format(i,image),end='')
			counter += 1
			i += 1
print("\nDone.")