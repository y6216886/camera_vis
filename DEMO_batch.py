from nerfvis import scene
import numpy as np
import sys
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from typing import Union
from math import cos,sin
import os
from colmap_utils import read_cameras_text,read_images_text,read_points3D_text,read_cameras_binary,read_images_binary,read_points3d_binary


#initialize scene
scene.set_title("My Scene")

# Set -y up camera space (opencv coordinates)
scene.set_opencv()





#read point cloud
point_cloud_path="/home/young/data/sacre_coeur/dense/fused.ply"
"read point cloud .ply file to numpy format"

import os
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
 
file_dir = point_cloud_path  #文件的路径
plydata = PlyData.read(file_dir)  # 读取文件
data = plydata.elements[0].data  # 读取数据
data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
data_np = np.zeros(data_pd.shape, dtype=np.float32)  # 初始化储存数据的array
property_names = data[0].dtype.names  # 读取property的名字
for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
    data_np[:, i] = data_pd[name]



point_cloud = data_np[:,:3]


# add points to scene
colors = data_np[:,-3:]/255.

scene.add_points("points", point_cloud, vert_color=colors)



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

if __name__ == '__main__':
    from PIL import Image
    scene.set_title("My Scene")

    # Set -y up camera space (opencv coordinates)
    scene.set_opencv()

    root_path = '/home/young/data/sacre_coeur/dense/sparse'

    cameras, images, points3D=read_model(root_path,'.bin')

    w2c_mats=[]
    c2w_mats=[]
    RDF_BRU = np.array([[0,1,0],[0,0,-1],[-1,0,0]])#BRU = RDF

    bottom = np.array([0,0,0,1.]).reshape([1,4])

    i=0
   #  breakpoint()
    Image_np_list=[]
    Rlist=[]
    tlist=[]
    for i in images:
        width, height= cameras[i].width, cameras[i].height
        f= 1.2*min(height, width) 
        print(i,'imgid')

        if i==1170: break
        R=images[i].qvec2rotmat()#RDF坐标系

        t=images[i].tvec.reshape(-1,1)
        
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        m=torch.Tensor(np.linalg.inv(m))

        w2c_mats.append(m)
        c2w_m=m
        c2w_mats.append(c2w_m)
        Rlist.append(c2w_m[:3,:3])
        tlist.append(c2w_m[:3,3].reshape(1,-1))

        imagepath=os.path.join('/home/young/data/sacre_coeur/dense/images', images[i].name)
        image_vec=Image.open(imagepath)
        Image_np=np.array(image_vec)/255.
        Image_np_list.append(Image_np)

    Rlist=np.stack(Rlist,0)
    tlist=np.concatenate(tlist,0)
    #add image and camera to scene
    scene.add_images(
                            f"images/i",
                            Image_np_list, # Can be a list of paths too (requires joblib for that) 
                            r=Rlist,
                            t=tlist,
                            # Alternatively: from nerfvis.utils import split_mat4; **split_mat4(c2ws)
                            focal_length=f,
                            z=0.3,
                            with_camera_frustum=True,
                        )
          
    scene.add_axes()
    scene.display()