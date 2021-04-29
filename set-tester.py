import urllib.request
import json
import os
from pathlib import Path
import operator

f = open('unique-artwork-20210413091556.json')
data = json.load(f)
f.close()

ordered_colors = {"BG":"BG","BGRUW":"WUBRG","BGUW":"WUBG","BRU":"UBR","BU":"UB","GRU":"URG","GU":"UG","R":"R","RW":"WR","W":"W","BGR":"BRG","BGRW":"WBRG","BGW":"WBG","BRUW":"WUBR","BUW":"WUB","G":"G","GRUW":"WURG","GUW":"WUG","RU":"UR","U":"U","B":"B","BGRU":"UBRG","BGU":"UBG","BR":"BR","BRW":"WBR","BW":"WB","GR":"RG","GRW":"WRG","GW":"WG","RUW":"WUR","UW":"WU"}
# counts = red(instant,sorc...) blue*
counts = dict()

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
		if i['type_line'].split(' ')[0] == 'Legendary' or i['type_line'].split(' ')[0] == 'Snow' or i['type_line'].split(' ')[0] == 'World' or i['type_line'].split(' ')[0] == 'Tribal' or i['type_line'].split(' ')[0] == 'Basic':
			#print('{}, {}'.format(c_i,i['type_line'].split(' ')[1]))
			counts[(c_i,(i['type_line'].split(' ')[1]))] = counts.get((c_i,(i['type_line'].split(' ')[1])),0) + 1
		else:
			#print('{}, {}'.format(c_i,i['type_line'].split(' ')[0]))
			counts[(c_i,(i['type_line'].split(' ')[0]))] = counts.get((c_i,(i['type_line'].split(' ')[0])),0) + 1

sorted_dict = dict(sorted(counts.items(), key=operator.itemgetter(0)))
curr = ''
for key in sorted_dict:
	if len(key[0]) < 2 and key[0]!='C':
		if key[0]!=curr:
			curr=key[0]
			print("\n--{}--".format(curr))
		print("{:<10s}\t{}".format(key[1],sorted_dict.get(key)))
		# # Make path
		# if not os.path.exists("img_raw/" + c_i):
		# 	os.makedirs("img_raw/" + c_i)

		# #get image name, create PATH to save to
		# # (this time saving set name as well, to allow multiple arts for one card)
		# IMAGE = "{}_{}".format(i['name'],i['set']).replace(' ','_').replace('/','')
		# PATH = "img_raw/" + c_i + "/" + IMAGE + ".jpg"

		# #download to proper directory, if it hasn't already been:
		# if not (Path(PATH).exists() and Path(PATH).is_file()):
		# 	print("downloading {} to {}".format(IMAGE,PATH))
		# 	urllib.request.urlretrieve(i['image_uris']['art_crop'],PATH)
		# else:
		# 	print("already downloaded, skipping "+IMAGE)
