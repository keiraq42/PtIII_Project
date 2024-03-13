import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
from PIL import Image, ImageEnhance

SampleWidth = 100
SampleDepth = 100

j_offset = 250

CellSize = 1

SeedNumber = 100

MaterialNumber = 4  # The number of materials within the bitmap
OtherMaterials = 1  # The number of materials not included in the bitmap (e.g. the projectile/container material)

cmc_list = []

for mat in range(MaterialNumber + 1):
    cmc_list.append("cmc{}".format(mat + 1))

dict = {}

Minimum_color = 0.2  # 0 = Black
Maximum_color = 0.8  # 1 = White


def generate_colors(length):
    colors = []
    for i in range(length):
        red = ((Maximum_color - Minimum_color) / (length - 1)) * i + Minimum_color
        green = ((Maximum_color - Minimum_color) / (length - 1)) * i + Minimum_color
        blue = ((Maximum_color - Minimum_color) / (length - 1)) * i + Minimum_color
        colors.append((red, green, blue))
    return colors


def voronoi_finite_polygons_2d_stretch(vor, radius=None, stretch_factor=10):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            # Stretch vertically
            far_point[1] += (far_point[1] - vor.points[p1][1]) * (stretch_factor - 1)

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

x_width = SampleWidth
y_width = SampleDepth

seeds = SeedNumber
points = np.random.rand(seeds, 2)

points[:, 0] *= x_width
points[:, 1] *= y_width

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, aspect='equal')

stretch_factor = 1  # Adjust the stretch factor as needed
vor = Voronoi(points)
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='k', line_width=0, line_alpha=1, point_size=0)

ax.set_xlim([0, x_width])
ax.set_ylim([0, y_width/stretch_factor])

ax.axis('off')

regions, vertices = voronoi_finite_polygons_2d_stretch(vor, stretch_factor=stretch_factor)

colours = generate_colors(MaterialNumber)

for region in regions:
    colour_ind = random.randint(0, len(colours) - 1)
    colour = colours[colour_ind]
    polygon = vertices[region]
    plt.fill(*zip(*polygon), c=colour, alpha=1)

fig.tight_layout()
fig.savefig('01_Voronoi.png')

ImageWidth = int(SampleWidth/CellSize)
ImageDepth = int(SampleDepth/CellSize)

def make_bitmap(image, output_width, output_height):
    img = Image.open(image)
    width, height = img.size
    left, top, right, bottom = width, height, 0, 0

    # Now crop the .png image to remove white space
    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))

            if pixel != (255, 255, 255, 255):  # Modify the condition if the white color has a different RGB value
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)
    cropped_img = img.crop((left, top, right + 1, bottom + 1))

    enhancer = ImageEnhance.Sharpness(cropped_img)
    enhancer.enhance(1)

    resized_img = cropped_img.resize((output_width, output_height))

    resized_img_1D = np.array(resized_img).flatten()
    histogram = plt.figure()
    histogram_axes = histogram.add_subplot(111)
    histogram_axes.hist(resized_img_1D, bins=16)
    histogram.savefig("03_Bitmap_Histogram.png")

    return resized_img

cropped_image = make_bitmap('01_Voronoi.png', ImageWidth, ImageDepth)
cropped_image.save("02_Voronoi.bmp")

def modify_bitmap(image):
    img = Image.open(image)
    img = img.convert('L')

    #img = np.array(img)
    #img = scipy.ndimage.grey_opening(img, size=(3,3))
    #img = Image.fromarray(img

    segmented_img = img.convert('P', palette=Image.ADAPTIVE, colors=MaterialNumber)
    return segmented_img

final_bitmap = modify_bitmap('02_Voronoi.bmp')
final_bitmap.save('04_FinalBitmap.bmp')

def export_mesh(filename, bitmap_name):

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

export_mesh("meso_m.iSALE", final_bitmap)
