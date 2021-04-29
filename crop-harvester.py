import urllib.request
import json
import os
from pathlib import Path

f = open('unique-artwork-20210413091556.json')
data = json.load(f)
f.close()

ordered_colors = {"BG":"BG","BGRUW":"WUBRG","BGUW":"WUBG","BRU":"UBR","BU":"UB","GRU":"URG","GU":"UG","R":"R","RW":"WR","W":"W","BGR":"BRG","BGRW":"WBRG","BGW":"WBG","BRUW":"WUBR","BUW":"WUB","G":"G","GRUW":"WURG","GUW":"WUG","RU":"UR","U":"U","B":"B","BGRU":"UBRG","BGU":"UBG","BR":"BR","BRW":"WBR","BW":"WB","GR":"RG","GRW":"WRG","GW":"WG","RUW":"WUR","UW":"WU"}

for i in data:
	# normal layout, normal art (should make shapes similar), legal in vintage (filter out "funny" and etc.)
	if i['layout'] == 'normal' and i['full_art'] == False and (i['legalities']['vintage'] == 'legal' or i['legalities']['vintage'] == 'restricted'):
		c_i = ''		# Get ordered colors for folder name
		if len(i['color_identity']) == 0:
			c_i = 'C'
		for n in i['color_identity']:
			c_i = (c_i + n)
		if (len(c_i) > 1):
			c_i = ordered_colors[c_i]

		# Make path

		# # get image name, create PATH to save to
		# # (this time saving set name as well, to allow multiple arts for one card)
		# IMAGE = "{}_{}".format(i['name'],i['set']).replace(' ','_').replace('/','')
		# PATH = "img_raw/" + c_i + "/" + IMAGE + ".jpg"

		if len(c_i) == 1 and c_i != 'C':
			# B mode: save creature, enchantment, instant, sorcery, land
			tp = None
			for t in ['Creature','Enchantment','Instant','Sorcery','Land']:
				if t in i['type_line']:
					tp = t
			
			if tp is not None:
				if not os.path.exists("imgs_sorted/" + c_i):
					os.makedirs("imgs_sorted/" + c_i)
				if not os.path.exists("imgs_sorted/" + c_i + "/" + tp):
					os.makedirs("imgs_sorted/" + c_i + "/" + tp)

				IMAGE = "{}_{}".format(i['name'],i['set']).replace(' ','_').replace('/','')
				PATH = "imgs_sorted/" + c_i + "/" + tp + "/" + IMAGE + ".jpg"


				#download to proper directory, if it hasn't already been:
				if not (Path(PATH).exists() and Path(PATH).is_file()):
					print("downloading {} to {}".format(IMAGE,PATH))
					urllib.request.urlretrieve(i['image_uris']['art_crop'],PATH)
				else:
					print("already downloaded, skipping "+IMAGE)
