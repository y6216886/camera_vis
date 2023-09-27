from nerfvis import scene
import numpy as np
import sys
sys.path.append("/home/young/code/large-scale-instant-neus/datasets")
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from typing import Union
# from preprocess.colmap_read_model import read_model
from math import cos,sin
from pyquaternion import Quaternion
import os
from colmap_utils import read_cameras_text,read_images_text,read_points3D_text,read_cameras_binary,read_images_binary,read_points3d_binary



scene.set_title("My Scene")

# Set -y up camera space (opencv coordinates)
scene.set_opencv()

# Alt set y up camera space (opengl coordinates, default)
#scene.set_opengl()



f = 1111.0
images = np.random.rand(1, 800, 800, 3)
c2ws = np.eye(4)[None]
point_cloud = np.random.randn(10000, 3) * 0.1
point_cloud_errs = np.random.rand(10000)

def _read_camera_dict(colmap_cameras, colmap_images):
    camera_dict = {}
    for image_id in colmap_images:
        image = colmap_images[image_id]

        img_name = image.name
        cam = colmap_cameras[image.camera_id]

        img_size = [cam.width, cam.height]
        params = list(cam.params)
        qvec = list(image.qvec)
        tvec = list(image.tvec)

        # w, h, fx, fy, cx, cy, s, qvec, tvec
        # camera_dict[img_name] = img_size + params + qvec + tvec
        breakpoint()
        fx, fy, cx, cy, s = params
        K = np.eye(4)
        K[0, 0] = fx
        K[0, 1] = s
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy

        rot = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        W2C = np.eye(4)
        W2C[:3, :3] = rot
        W2C[:3, 3] = np.array(tvec)
        C2W = np.linalg.inv(W2C)
        camera_dict[img_name] = {
            'K': K.flatten().tolist(),
            'W2C': W2C.flatten().tolist(),
            'C2W': C2W.flatten().tolist(),
            'img_size': img_size
        }
        
    return camera_dict




camera_scale=0
def normalize(x):
    return x / torch.linalg.norm(x)
# 需要输入为[num,3,4]的tensor/ndarray poses
def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D

def Rx(theta):
    R = torch.eye(3)
    R[1,1] = cos(theta)
    R[2,2] = cos(theta)
    R[1,2] = -sin(theta)
    R[2,1] = sin(theta)
    return R
def Ry(theta):
    R = torch.eye(3)
    R[0,0] = cos(theta)
    R[2,2] = cos(theta)
    R[0,2] = -sin(theta)
    R[2,0] = sin(theta)
    return R

def draw_sheet(x:Tensor,y:Tensor,sheet_type='s'):
    plt.figure()
    if x.shape[0]>1:
        idx = torch.randint(x.shape[0],(1,))
    else:
        idx = 0
    x=x.cpu().detach().numpy()
    y=y.cpu().detach().numpy()
    # plt.scatter(x[idx,...],y[idx,...])
    plt.scatter(x.mean(axis=0),y.mean(axis=0))
    # plt.savefig(f'weight_{sheet_type}.png')
    plt.savefig(f'weight_t.png')

def draw_poses(poses_:Union[Tensor,ndarray]=None,
               rays_o_=None,
               rays_d_=None,
               pts3d:ndarray = None,
               aabb_=None,
               aabb_idx=None,
               img_wh=None,
               t_min=None,
               t_max=None
               )->None:
    if isinstance(poses_,Tensor):         
        poses=poses_[None,:,:].to("cpu").numpy()
    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')

    # ax = fig.add_subplot(projection='3d')
    if poses_ is not None:
        poses=poses_
        
        try:
            poses = poses.to("cpu")
        except:
            pass
        if len(poses_.shape)<=2:
            poses = poses[None,:,:]
        # for i in range(0,poses.shape[0]):
        for i in range(0,poses.shape[0],int(poses.shape[0]/200)):
            # breakpoint()
            center_camera=poses[i,:3,3:4]
            xyz_camera=center_camera+poses[i,:3,:3]*0.1
            ax.scatter(center_camera[0,0],center_camera[1,0],center_camera[2,0],cmap="Reds")
            ax.plot([center_camera[0,0],xyz_camera[0,0]],[center_camera[1,0],xyz_camera[1,0]],[center_camera[2,0],xyz_camera[2,0]],color='r')
            ax.plot([center_camera[0,0],xyz_camera[0,1]],[center_camera[1,0],xyz_camera[1,1]],[center_camera[2,0],xyz_camera[2,1]],color='g')
            ax.plot([center_camera[0,0],xyz_camera[0,2]],[center_camera[1,0],xyz_camera[1,2]],[center_camera[2,0],xyz_camera[2,2]],color='b')

            ax.plot([xyz_camera[0,1],xyz_camera[0,0]],[xyz_camera[1,1],xyz_camera[1,0]],[xyz_camera[2,1],xyz_camera[2,0]],color='m')
            ax.plot([xyz_camera[0,2],xyz_camera[0,1]],[xyz_camera[1,2],xyz_camera[1,1]],[xyz_camera[2,2],xyz_camera[2,1]],color='m')
            ax.plot([xyz_camera[0,0],xyz_camera[0,2]],[xyz_camera[1,0],xyz_camera[1,2]],[xyz_camera[2,0],xyz_camera[2,2]],color='m')

            ax.scatter([center_camera[0,0]],[center_camera[1,0]],[center_camera[2,0]],color='m')
    if pts3d is not None:
        try:
            pts3d = pts3d.cpu()
        except:
            pass
        for i in range(0,pts3d.shape[0],int(pts3d.shape[0]/300)):
            ax.scatter([pts3d[i,0]],[pts3d[i,1]],[pts3d[i,2]],color='b')
    if rays_o_ is not None:
        rays_o = rays_o_.to("cpu")
        rays_d = rays_d_.to("cpu")
        
        if t_min is not None:
            t_max_ = t_max.to("cpu")
            t_min_ = t_min.to("cpu")
            rays = rays_o+rays_d*t_max_.view(-1,1)
            # rays = rays_o+rays_d*t_min_.view(-1,1)
        else:
            rays = rays_o+rays_d*1
        ax.scatter([rays_o[0,0]],[rays_o[0,1]],[rays_o[0,2]],color='r')
        
        for i in range(0,rays_d.shape[0],int(rays_d.shape[0]/100)):
        # for i in range(0,rays_d.shape[0]):
            ax.plot([rays_o[0,0],rays[i,0]],[rays_o[0,1],rays[i,1]],[rays_o[0,2],rays[i,2]],color='b')
        
        
        # ax.plot([rays_o[0,0],rays[0,0]],[rays_o[0,1],rays[0,1]],[rays_o[0,2],rays[0,2]],color='b')
        # ax.plot([rays_o[0,0],rays[img_wh[0]-1,0]],[rays_o[0,1],rays[img_wh[0]-1,1]],[rays_o[0,2],rays[img_wh[0]-1,2]],color='b')
        # ax.plot([rays_o[0,0],rays[img_wh[0]*(img_wh[1]-1),0]],[rays_o[0,1],rays[img_wh[0]*(img_wh[1]-1),1]],[rays_o[0,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        # ax.plot([rays_o[0,0],rays[-1,0]],[rays_o[0,1],rays[-1,1]],[rays_o[0,2],rays[-1,2]],color='b')
            
        # ax.plot([rays[0,0],rays[img_wh[0]-1,0]],[rays[0,1],rays[img_wh[0]-1,1]],[rays[0,2],rays[img_wh[0]-1,2]],color='b')
        # ax.plot([rays[0,0],rays[img_wh[0]*(img_wh[1]-1),0]], [rays[0,1],rays[img_wh[0]*(img_wh[1]-1),1]], [rays[0,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        # ax.plot([rays[-1,0],rays[img_wh[0]-1,0]],[rays[-1,1],rays[img_wh[0]-1,1]],[rays[-1,2],rays[img_wh[0]-1,2]],color='b')
        
        # ax.plot([rays[-1,0],rays[img_wh[0]*(img_wh[1]-1),0]],[rays[-1,1],rays[img_wh[0]*(img_wh[1]-1),1]],[rays[-1,2],rays[img_wh[0]*(img_wh[1]-1),2]],color='b')
        
    if aabb_ is not None:
        for i in range(0,aabb_.shape[0]):
            color = 'r'  
            zorder = 0   
            if aabb_idx is not None:    
                if i in aabb_idx:
                    color = 'b'
                    zorder = 1        
            aabb = aabb_[i].to("cpu")
            ax.plot([aabb[0],aabb[3]],[aabb[1],aabb[1]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[1],aabb[4]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[1],aabb[1]],[aabb[2],aabb[5]],color=color,zorder=zorder)
            
            ax.plot([aabb[3],aabb[0]],[aabb[1],aabb[1]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[1],aabb[4]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[1],aabb[1]],[aabb[5],aabb[2]],color=color,zorder=zorder)
            
            ax.plot([aabb[0],aabb[3]],[aabb[4],aabb[4]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[4],aabb[1]],[aabb[5],aabb[5]],color=color,zorder=zorder)
            ax.plot([aabb[0],aabb[0]],[aabb[4],aabb[4]],[aabb[5],aabb[2]],color=color,zorder=zorder)
            
            ax.plot([aabb[3],aabb[0]],[aabb[4],aabb[4]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[4],aabb[1]],[aabb[2],aabb[2]],color=color,zorder=zorder)
            ax.plot([aabb[3],aabb[3]],[aabb[4],aabb[4]],[aabb[2],aabb[5]],color=color,zorder=zorder)
        
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([-5, 5])
    plt.xlabel('x')
    plt.ylabel('y')
    
    # plt.xticks(np.arange(-5, 5, 1))
    # plt.yticks(np.arange(-5, 5, 1))
    plt.autoscale(True)
    # plt.show()
    for i in range(0,20):
        ax.view_init(elev=10*i-100, azim=i*4)
        plt.savefig(f'./test{i}.png')
R_1 = torch.tensor([
[1.000000000000, 0.000000000000, 0.000000000000 ,0.000000000000],
[0.000000000000, 0.738635659218, -0.674104869366, 27.879625320435],
[0.000000000000, 0.674104869366, 0.738635659218 ,6.405389785767],
[0.000000000000, 0.000000000000, 0.000000000000 ,1.000000000000]
])
R_2 = torch.tensor([
[1.000000000000, 0.000000000000, 0.000000000000, 56.128658294678],
[0.000000000000, 1.000000000000, 0.000000000000, -20.065317153931],
[0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000],
[0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]
])
if __name__ == '__main__':
    from PIL import Image
    scene.set_title("My Scene")

    # Set -y up camera space (opencv coordinates)
    scene.set_opencv()

    root_path = '/home/young/data/sacre_coeur/dense/sparse'

    cameras, images, points3D=read_model(root_path,'.bin')
    camera_dict=_read_camera_dict(cameras, images)
    w2c_mats=[]
    c2w_mats=[]
    RDF_BRU = np.array([[0,1,0],[0,0,-1],[-1,0,0]])#BRU = RDF
    # ppp = np.array([0,0,1]).reshape([1,3])
    bottom = np.array([0,0,0,1.]).reshape([1,4])
    # R_h = np.concatenate([np.concatenate([RDF_BRU, np.zeros([3,1])], 1), bottom], 0)
    R_correct = R_2 @ R_1
    # R_correct = Ry(0)
    i=0
   #  breakpoint()
    Image_np_list=[]
    Rlist=[]
    tlist=[]
    for i in images:
        print(i)
        # breakpoint()
        if i==1000: break
        
        C2W=camera_dict[images[i].name]['C2W']
        breakpoint()
        imagepath=os.path.join('/home/young/data/sacre_coeur/dense/images', images[i].name)
        image_vec=Image.open(imagepath)
        Image_np=np.array(image_vec)/255.
        Image_np_list.append(Image_np)
        # for i in range(len(c2ws)):
        # breakpoint()
    # breakpoint()
    # Image_np_list=np.concatenate(Image_np_list,0)
    Rlist=np.stack(Rlist,0)
    tlist=np.concatenate(tlist,0)
    scene.add_images(
                            f"images/i",
                            Image_np_list, # Can be a list of paths too (requires joblib for that) 
                            r=Rlist,
                            t=tlist,
                            # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
                            focal_length=f,
                            z=0.5,
                            with_camera_frustum=True,
                        )
            # r: c2w rotation (N, 3, 3) or (N, 4) etc
            # t: c2w translation (N, 3,)
            # focal_length: focal length (in pixels, real image size as loaded will be used)
            # z: size of camera
    # pts3d=[]
    # for point in points3D:#RDF下的三维点世界坐标(以第一帧相机坐标为世界坐标)，左乘P^-1得到BRU世界坐标
    #     pt3d = torch.concat([torch.tensor(points3D[point].xyz),torch.tensor([1])],dim=-1).view(4,1).float()
    #     # pts3d.append((torch.linalg.inv(R_correct) @ pt3d).reshape(1,4)[:,:3])
    #     pts3d.append((R_correct @ pt3d).reshape(1,4)[:,:3])
    # pts3d = torch.tensor(np.concatenate(pts3d,0))
    # c2w_mats = torch.stack(c2w_mats)
    
    # # np.linalg.inv(R_h) @ c2w_mats @ R_h
    # # ppp = RDF2BRU @ ppp
    # # c2w_mats=c2w_mats.reshape(len(images),3,-1)
    # # w2c_mats=w2c_mats.reshape(len(images),4,-1)
    # # c2w_mats=np.concatenate(c2w_mats,axis=0)
    # # np.savetxt("/home/will/data/data_lab_reduce_frame/poses.txt",c2w_mats)
    # # draw_poses(pts3d=pts3d)
    # import trimesh
    # pcd = trimesh.PointCloud(pts3d.numpy())
    # # pcd.export('./check.ply')
    # draw_poses(c2w_mats)
    # # print(len(poses))
    # pass
    scene.add_axes()
    scene.display()