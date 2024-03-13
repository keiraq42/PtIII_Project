import numpy as np

from PIL import Image


MaterialNumber = 4 # The number of materials within the bitmap

OtherMaterials = 1 # The number of materials not included in the bitmap (e.g. the projectile/container material)


j_offset = 550 #State the offset of y-cells for the bitmap region to be within iSALE grid - measured from top or bottom?


cmc_list = []

for mat in range(MaterialNumber+1):

	cmc_list.append("cmc{}".format(mat+1))

	dict = {}


# Define Required Functions

def modify_bitmap(image):

#Reads a bmp image, converts it to appropriate format

	img = Image.open(image)


	print("Image Size = {}".format(img.size))


	img = img.convert('L')

	segmented_img = img.convert('P', palette=Image.ADAPTIVE, colors=MaterialNumber)

	return segmented_img


def export_mesh(filename, bitmap_name):

# exports bitmap as meso_m.iSALE file


	data=np.array(bitmap_name)


	with open(filename, "w") as f:

		f.write("{} {} \n".format(data.size, MaterialNumber+OtherMaterials))

		for i in range(data.shape[1]):
	
			for j in range(data.shape[0]):

				i_index = i

				j_index = j + j_offset

				vx = 0

				vy = 0

	
				f.write("{} {} {} {} ".format(j_index, i_index, vx, vy))
                                # Flip j_index with i_index


				for cmc in range(MaterialNumber):
	
					if data[j,i] == cmc:

						for mat in cmc_list:

							if mat == cmc_list[cmc+OtherMaterials]:

								dict[mat] = 1.0

							else:

								dict[mat] = 0.0


				for cmc in range(len(cmc_list)):

					f.write("{:f} ".format(dict[cmc_list[cmc]]))

				f.write("\n")


# Read, modify, then export bmp as meso_m.iSALE file

image = 'sample_crop.bmp'

image = modify_bitmap(image)

export_mesh("meso_m.iSALE", image)
