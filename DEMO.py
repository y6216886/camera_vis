from nerfvis import scene
import numpy as np

scene.set_title("My Scene")

# Set -y up camera space (opencv coordinates)
scene.set_opencv()

# Alt set y up camera space (opengl coordinates, default)
#scene.set_opengl()


# Example data
f = 1111.0
images = np.random.rand(1, 800, 800, 3)
c2ws = np.eye(4)[None]
point_cloud = np.random.randn(10000, 3) * 0.1
point_cloud_errs = np.random.rand(10000)

# To show errors as colors
colors = np.zeros_like(point_cloud)
colors[:, 0] = point_cloud_errs / point_cloud_errs.max()
scene.add_points("points", point_cloud, vert_color=colors)
# Else
# scene.add_points("points", point_cloud, color=[0.0, 0.0, 0.0])

for i in range(len(c2ws)):
   scene.add_images(
                 f"images/i",
                 images, # Can be a list of paths too (requires joblib for that) 
                 r=c2ws[:, :3, :3],
                 t=c2ws[:, :3, 3],
                 # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
                 focal_length=f,
                 z=0.5,
                 with_camera_frustum=True,
             )
   # r: c2w rotation (N, 3, 3) or (N, 4) etc
   # t: c2w translation (N, 3,)
   # focal_length: focal length (in pixels, real image size as loaded will be used)
   # z: size of camera

# Old way for reference
# scene.add_camera_frustum("cameras", r=c2ws[:, :3, :3], t=c2ws[:, :3, 3], focal_length=f,
#                         image_width=images.shape[2], image_height=images.shape[1],
#                         z=0.5, connect=False, color=[1.0, 0.0, 0.0])

# for i in range(len(c2ws)):
#    scene.add_image(
#                  f"images/i",
#                  images[i], # Can be path too
#                  r=c2ws[i, :3, :3],
#                  t=c2ws[i, :3, 3],
#                  focal_length=f,
#                  z=0.5)
#    # r: c2w rotation (3, 3)
#    # t: c2w translation (3,)
#    # focal_length: focal length (in pixels, real image size as loaded will be used)
#    # z: distance along z to place the camera
scene.add_axes()
scene.display()