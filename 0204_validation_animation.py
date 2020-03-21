import argparse
import json
import numpy as np
import os
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import sys
from tqdm import tqdm

from utils import preprocess, pooling, get_pooling_index
from utils import setup_meshes, split
from utils import loss_surf, loss_edge, loss_lap, loss_norm
from architectures import VGG as Encoder, G_Res_Net, MyResnet

import kaolin as kal
import pyredner
import math
import random

import imageio

from torch.utils.tensorboard import SummaryWriter

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
"""
Commandline arguments
"""

pyredner.set_use_gpu(torch.cuda.is_available())

parser = argparse.ArgumentParser()
parser.add_argument('-expid',
                    type=str,
                    default='0115_default',
                    help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories',
                    type=str,
                    nargs='+',
                    default=['plane'],
                    help='list of object classes to use')
parser.add_argument('-resolution',
                    type=int,
                    default=224,
                    help='resolution of image')
parser.add_argument('-num_samples',
                    type=int,
                    default=32,
                    help='rendering samples')
parser.add_argument('-epochs',
                    type=int,
                    default=50,
                    help='Number of train epochs.')
parser.add_argument('-batchsize', type=int, default=1, help='Batch size.')
parser.add_argument('-val_batchsize',
                    type=int,
                    default=32,
                    help='Batch size for validation.')
parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('-val-every',
                    type=int,
                    default=5,
                    help='Validation frequency (epochs).')
parser.add_argument('-print_every',
                    type=int,
                    default=1000,
                    help='Print frequency (batches).')
parser.add_argument('-logdir',
                    type=str,
                    default='log',
                    help='Directory to log data to.')
parser.add_argument('-save-model',
                    action='store_true',
                    help='Saves the model and a snapshot \
    of the optimizer state.')
parser.add_argument('-vertices_scale',
                    type=float,
                    default=1.0,
                    help='vertices_scale')
parser.add_argument('-surf_scale', type=int, default=3000, help='surf_scale')
parser.add_argument('-edge_scale', type=int, default=300, help='edge_scale')
parser.add_argument('-lap_scale', type=int, default=150, help='lap_scale')
parser.add_argument('-norm_scale', type=float, default=.5, help='norm_scale')
parser.add_argument('-material_num', type=int, default=7, help='material_num')

parser.add_argument(
    '-encoder_path',
    type=str,
    default=
    "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0118kabutogani2466_car_epoch100/model/best_encoder.pth",
    # "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0119megisu_2466_car_save_rand/model/encoder_epoch49.pth",
    help='encoder_path')
parser.add_argument(
    '-mesh0_path',
    type=str,
    default=
    "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0118kabutogani2466_car_epoch100/model/best_mesh_update_0.pth",
    # "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0119megisu_2466_car_save_rand/model/mesh_update_0_epoch49.pth",
    help='mesh0_path')
parser.add_argument(
    '-mesh1_path',
    type=str,
    default=
    "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0118kabutogani2466_car_epoch100/model/best_mesh_update_1.pth",
    # "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0119megisu_2466_car_save_rand/model/mesh_update_1_epoch49.pth",
    help='mesh1_path')
parser.add_argument(
    '-mesh2_path',
    type=str,
    default=
    "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0118kabutogani2466_car_epoch100/model/best_mesh_update_2.pth",
    # "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0119megisu_2466_car_save_rand/model/mesh_update_2_epoch49.pth",
    help='mesh2_path')
parser.add_argument(
    '-optimizer_path',
    type=str,
    default=
    "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0118kabutogani2466_car_epoch100/model/best_optim.pth",
    # "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0119megisu_2466_car_save_rand/model/recent_optim_epoch49.pth",
    help='optimizer_path')

parser.add_argument(
    '-material_estimator_path',
    type=str,
    default=
    "/home/mil/kasuga/kaolin_1217/kaolin/examples/ImageRecon/Pixel2Mesh_pyredner/log/0124jinbei_car_all_sample32_roughrange/model/material_estimator_epoch18.pth",
    help='optimizer_path')

args = parser.parse_args()

envmap_img = pyredner.imread('grace-new.exr')
envmap_img = torch.min(envmap_img, torch.tensor([1.0]))
envmap_img = envmap_img.to(pyredner.get_device())
envmap = pyredner.EnvironmentMap(envmap_img * 2.0)


class Points_Meshes_Dataset(torch.utils.data.Dataset):
    def __init__(self, points_set, meshes_set):
        self.points_set = points_set
        self.meshes_set = meshes_set
        self.data_num = len(points_set.names)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_points = self.points_set[idx]['data']['points']
        out_normals = self.points_set[idx]['data']['normals']
        out_verties = self.meshes_set[idx]['data']['vertices']
        out_faces = self.meshes_set[idx]['data']['faces']

        return out_points, out_normals, out_verties, out_faces


def my_collate_fn(batch):
    points_list = []
    normals_list = []
    vertices_list = []
    faces_list = []
    for sample in batch:
        points, normals, vertices, faces = sample
        points_list.append(points)
        normals_list.append(normals)
        vertices_list.append(vertices)
        faces_list.append(faces)

    points_list = torch.stack(points_list, dim=0)
    normals_list = torch.stack(normals_list, dim=0)
    # vertices_list = torch.stack(vertices_list, dim=0)
    # faces_list = torch.stack(faces_list, dim=0)

    return [points_list, normals_list, vertices_list, faces_list]


def make_obj_and_cam(vertices, faces, normals, diffuse_reflectance,
                     specular_reflectance, roughness, distance, theta, azimuth,
                     two_sided):

    material = pyredner.Material(
        diffuse_reflectance=diffuse_reflectance,
        specular_reflectance=specular_reflectance,
        roughness=roughness,
        two_sided=two_sided
    )  # two_sided = True is for dealing with bugs on shapenets

    obj = pyredner.Object(vertices=vertices,
                          indices=faces,
                          normals=normals,
                          material=material)

    cam0 = pyredner.automatic_camera_placement([obj],
                                               resolution=(args.resolution,
                                                           args.resolution))

    x = distance * math.sin(math.pi * theta / 180.0) * math.cos(
        math.pi * azimuth / 180.0)
    y = distance * math.cos(math.pi * theta / 180.0)
    z = distance * math.sin(math.pi * theta / 180.0) * math.sin(
        math.pi * azimuth / 180.0)
    cam = pyredner.Camera(position=cam0.look_at + torch.tensor([x, y, z]),
                          look_at=cam0.look_at,
                          up=cam0.up,
                          fov=cam0.fov,
                          resolution=(args.resolution, args.resolution))

    return obj, cam


def make_scenes(vertices_list,
                faces_list,
                azimuth_batch,
                diffuse_reflectance_batch,
                specular_reflectance_batch,
                roughness_batch,
                two_sided=False):
    batch_size = len(vertices_list)
    scene_list = []

    for index in range(batch_size):

        vertices = vertices_list[index]
        faces = faces_list[index].type(torch.int32)
        normals = pyredner.compute_vertex_normal(vertices, faces)

        # set materials at random
        diffuse_reflectance = diffuse_reflectance_batch[index]
        specular_reflectance = specular_reflectance_batch[index]
        roughness = roughness_batch[index]
        distance = 1.2
        theta = 60.0
        azimuth = azimuth_batch[index].item()

        obj, cam = make_obj_and_cam(vertices, faces, normals,
                                    diffuse_reflectance, specular_reflectance,
                                    roughness, distance, theta, azimuth,
                                    two_sided)
        scene = pyredner.Scene(objects=[obj], camera=cam, envmap=envmap)
        scene_list.append(scene)

        material = pyredner.Material(
            diffuse_reflectance=diffuse_reflectance,
            specular_reflectance=specular_reflectance,
            roughness=roughness,
            two_sided=two_sided
        )  # two_sided = True is for dealing with bugs on shapenets

        # obj = pyredner.Object(vertices=vertices,
        #                       indices=faces,
        #                       normals=None,
        #                       material=material)
        # pyredner.save_obj(
        #     obj, os.path.join(args.logdir, 'obj/batch{}.obj'.format(index)))

    return scene_list


"""
Dataset
"""

# points_set = kal.dataloader.ShapeNet.Points(root ='../../datasets/',categories =args.categories , \
# 	download = True, train = True, split = .7, num_points=2466 )
# images_set = kal.dataloader.ShapeNet.Images(root ='../../datasets/',categories =args.categories , \
# 	download = True, train = True,  split = .7, views=23, transform= preprocess )
# train_set = kal.dataloader.ShapeNet.Combination([points_set, images_set], root='../../datasets/')

points_set = kal.datasets.shapenet.ShapeNet_Points(root ='/data/umihebi0/users/kasuga/car_shapenet',cache_dir='/data/umihebi0/users/kasuga/car_shapenet/cache/train_2466', categories =args.categories , \
 train = True, split = .7, num_points=2466)
meshes_set = kal.datasets.shapenet.ShapeNet_Meshes(root ='/data/umihebi0/users/kasuga/car_shapenet',categories =args.categories , \
 train = True,  split = .7)
train_set = Points_Meshes_Dataset(points_set, meshes_set)

dataloader_train = DataLoader(train_set,
                              batch_size=1,
                              shuffle=True,
                              num_workers=8,
                              collate_fn=my_collate_fn)

# points_set_valid = kal.dataloader.ShapeNet.Points(root ='../../datasets/',categories =args.categories , \
# 	download = True, train = False, split = .7, num_points=10000 )
# images_set_valid = kal.dataloader.ShapeNet.Images(root ='../../datasets/',categories =args.categories , \
# 	download = True, train = False,  split = .7, views=1, transform= preprocess )
# valid_set = kal.dataloader.ShapeNet.Combination([points_set_valid, images_set_valid], root='../../datasets/')

# dataloader_val = DataLoader(valid_set, batch_size=1, shuffle=False,
# 	num_workers=8)
points_set_valid = kal.datasets.shapenet.ShapeNet_Points(root ='/data/umihebi0/users/kasuga/car_shapenet',cache_dir='/data/umihebi0/users/kasuga/car_shapenet/cache/valid', categories =args.categories , \
 train = False, split = .7, num_points=5000)
meshes_set_valid = kal.datasets.shapenet.ShapeNet_Meshes(root ='/data/umihebi0/users/kasuga/car_shapenet',categories =args.categories , \
 train = False,split = .7)
train_set_valid = Points_Meshes_Dataset(points_set_valid, meshes_set_valid)

dataloader_valid = DataLoader(train_set_valid,
                              batch_size=1,
                              shuffle=False,
                              num_workers=8,
                              collate_fn=my_collate_fn)
"""
Model settings
"""
meshes = setup_meshes(filename='meshes/156.obj', device=pyredner.get_device())

encoder = Encoder().to(pyredner.get_device())
encoder.load_state_dict(torch.load(args.encoder_path))

mesh_update_kernels = [963, 1091, 1091]
mesh_updates = [
    G_Res_Net(mesh_update_kernels[i], hidden=128,
              output_features=3).to(pyredner.get_device()) for i in range(3)
]
mesh_updates[0].load_state_dict(torch.load(args.mesh0_path))
mesh_updates[1].load_state_dict(torch.load(args.mesh1_path))
mesh_updates[2].load_state_dict(torch.load(args.mesh2_path))

material_estimator = MyResnet(args.material_num).to(pyredner.get_device())

material_estimator.load_state_dict(torch.load(args.material_estimator_path))

parameters = list(material_estimator.parameters())
optimizer = optim.Adam(parameters, lr=args.lr)

encoding_dims = [56, 28, 14, 7]
"""
Initial settings
"""

# Create log directory, if it doesn't already exist
args.logdir = os.path.join(args.logdir, args.expid)
if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)
    save_train_path = os.path.join(args.logdir, 'train')
    save_valid_path = os.path.join(args.logdir, 'valid')
    save_obj_path = os.path.join(args.logdir, 'obj')
    save_model_path = os.path.join(args.logdir, 'model')
    os.makedirs(save_train_path)
    os.makedirs(save_valid_path)
    os.makedirs(save_obj_path)
    os.makedirs(save_model_path)
    print('Created dir:', args.logdir)
    print('Created dir:', save_train_path)
    print('Created dir:', save_valid_path)
    print('Created dir:', save_obj_path)
    print('Created dir:', save_model_path)

# Log all commandline args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
log_file_name = os.path.join(args.logdir, 'train_log')


def compute_camera_params(azimuth, elevation, distance):
    def vector_cross(vec1, vec2):
        x = vec1[1] * vec2[2] - vec1[2] * vec2[1]
        y = vec1[2] * vec2[0] - vec1[0] * vec2[2]
        z = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return torch.stack([x, y, z], dim=0)

    def vector_normalize(vec):
        length = torch.sqrt(torch.sum(vec * vec, dim=-1, keepdim=True))
        return vec / (length + 1e-12)

    # theta = np.deg2rad(azimuth)
    # phi = np.deg2rad(elevation)
    theta = azimuth * math.pi / 180.0
    phi = torch.tensor([elevation * math.pi / 180.0],
                       device=pyredner.get_device())

    camY = distance * torch.sin(phi)
    temp = distance * torch.cos(phi)
    camX = temp * torch.cos(theta)
    camZ = temp * torch.sin(theta)

    cam_pos = torch.cat([camX, camY, camZ], dim=0)

    axisZ = cam_pos.clone()
    axisY = torch.tensor([0, 1, 0], device=pyredner.get_device())
    axisX = vector_cross(axisY, axisZ)
    axisY = vector_cross(axisZ, axisX)

    axisX = vector_normalize(axisX)
    axisY = vector_normalize(axisY)
    axisZ = vector_normalize(axisZ)

    cam_mat = torch.stack([axisX, axisY, axisZ], dim=0)
    # l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    # l2[l2 == 0] = 1
    # cam_mat = cam_mat / np.expand_dims(l2, 1)

    return cam_mat, cam_pos


class Engine(object):
    """Engine that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.

    """
    def __init__(self, cur_epoch=0, print_every=1, validate_every=1):
        self.cur_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1000.
        self.rendering_val_loss = []
        self.rendering_bestval = 1000.

    def validate(self):
        encoder.eval(), [m.eval() for m in mesh_updates]
        with torch.no_grad():
            inp_list = []
            out_list = []

            for i in range(args.val_batchsize):
                data = iter(dataloader_train).next()

                # data creation
                tgt_points_batch = data[0].to(
                    pyredner.get_device())  #もとからtensor
                tgt_normals_batch = data[1].to(
                    pyredner.get_device())  #もとからtensor
                inp_vertices_list = data[2]  #.to(args.device) #もとはlist
                inp_faces_list = data[3]  #.to(args.device) #もとはlist

                # batchsizeはちょうど割り切れるとは限らないので，args.batchsizeではなくあくまでdataのsizeにしておく
                # azimuth_batch = torch.rand(data[0].shape[0],
                #                            device=pyredner.get_device()) * 360.0
                azimuth_batch = torch.randint(
                    0,
                    24, (data[0].shape[0], ),
                    dtype=torch.float32,
                    device=pyredner.get_device()) * 15.0

                inp_diffuse_reflectance_batch = torch.rand(
                    [data[0].shape[0], 3], device=pyredner.get_device())
                inp_specular_reflectance_batch = torch.rand(
                    [data[0].shape[0], 3], device=pyredner.get_device())
                inp_roughness_batch = torch.rand(
                    [data[0].shape[0], 1],
                    device=pyredner.get_device()) * 0.2 + 1e-5

                inp_scene_list = make_scenes(inp_vertices_list, inp_faces_list,
                                             azimuth_batch,
                                             inp_diffuse_reflectance_batch,
                                             inp_specular_reflectance_batch,
                                             inp_roughness_batch, True)

                inp_image_batch = pyredner.render_pathtracing(
                    scene=inp_scene_list, num_samples=128)
                inp_image_batch = inp_image_batch.permute(0, 3, 1, 2)

                inp_image_batch = torch.pow(
                    torch.max(
                        inp_image_batch,
                        torch.tensor([0.0], device=pyredner.get_device())),
                    1.0 / 2.2)
                # input azimuth
                azimuth_features = (
                    azimuth_batch - 180.0
                ) / 180.0  # camera features [0.0, 360.0] -> [-1.0, 1.0]
                azimuth_features = azimuth_features.unsqueeze(
                    1)  # [batchsize, 1]

                # cam_mat, cam_pos = kal.mathutils.geometry.transformations.compute_camera_params(azimuth_batch[0], 60.0, 1.2)
                cam_mat, cam_pos = compute_camera_params(
                    azimuth_batch[0], 60.0, 1.2)
                # cam_mat = cam_mat.to(pyredner.get_device())
                # cam_pos = cam_pos.to(pyredner.get_device())

                ###############################
                ########## inference ##########
                ###############################
                img_features = encoder(inp_image_batch)

                ##### layer_1 #####
                pool_indices = get_pooling_index(meshes['init'][0].vertices,
                                                 cam_mat, cam_pos,
                                                 encoding_dims)
                projected_image_features = pooling(img_features, pool_indices)
                full_vert_features = torch.cat(
                    (meshes['init'][0].vertices, projected_image_features),
                    dim=1)

                pred_verts, future_features = mesh_updates[0](
                    full_vert_features, meshes['adjs'][0])
                meshes['update'][0].vertices = pred_verts.clone()

                ##### layer_2 #####
                future_features = split(meshes, future_features, 0)
                pool_indices = get_pooling_index(meshes['init'][1].vertices,
                                                 cam_mat, cam_pos,
                                                 encoding_dims)
                projected_image_features = pooling(img_features, pool_indices)
                full_vert_features = torch.cat(
                    (meshes['init'][1].vertices, projected_image_features,
                     future_features),
                    dim=1)

                pred_verts, future_features = mesh_updates[1](
                    full_vert_features, meshes['adjs'][1])
                meshes['update'][1].vertices = pred_verts.clone()

                ##### layer_3 #####
                future_features = split(meshes, future_features, 1)
                pool_indices = get_pooling_index(meshes['init'][2].vertices,
                                                 cam_mat, cam_pos,
                                                 encoding_dims)
                projected_image_features = pooling(img_features, pool_indices)
                full_vert_features = torch.cat(
                    (meshes['init'][2].vertices, projected_image_features,
                     future_features),
                    dim=1)

                pred_verts, future_features = mesh_updates[2](
                    full_vert_features, meshes['adjs'][2])
                meshes['update'][2].vertices = pred_verts.clone()

                # material estimation
                out_materials_batch = material_estimator(
                    inp_image_batch, azimuth_features)

                out_diffuse_reflectance_batch = out_materials_batch[:, 0:3]
                out_specular_reflectance_batch = out_materials_batch[:, 3:6]
                out_roughness_batch = out_materials_batch[:, 6].unsqueeze(1)
                # out_specular_reflectance_batch = inp_specular_reflectance_batch
                # out_roughness_batch = inp_roughness_batch  # roughnessはそのまま

                out_scene_list = make_scenes(
                    meshes['update'][2].vertices.unsqueeze(0),
                    meshes['update'][2].faces.unsqueeze(0), azimuth_batch,
                    out_diffuse_reflectance_batch,
                    out_specular_reflectance_batch, out_roughness_batch, True)

                out_image_batch = pyredner.render_pathtracing(
                    scene=out_scene_list, num_samples=128)
                out_image_batch = out_image_batch.permute(0, 3, 1, 2)

                out_image_batch = torch.clamp(out_image_batch,
                                              min=0.00001,
                                              max=1.0)
                out_image_batch = torch.pow(out_image_batch, 1.0 / 2.2)

                out_list.append(out_image_batch)

                loop = tqdm(list(range(0, 360, 4)))
                writer = imageio.get_writer(os.path.join(
                    args.logdir,
                    'valid/rotation_epoch{}_{}.gif'.format(self.cur_epoch, i)),
                                            mode='i')
                for num, azimuth in enumerate(loop):
                    # rest mesh to initial state
                    loop.set_description('Drawing rotation')
                    azimuth_batch = torch.ones([
                        data[0].shape[0],
                    ],
                                               dtype=torch.float32) * azimuth
                    out_scene_list = make_scenes(
                        meshes['update'][2].vertices.unsqueeze(0),
                        meshes['update'][2].faces.unsqueeze(0), azimuth_batch,
                        out_diffuse_reflectance_batch,
                        out_specular_reflectance_batch, out_roughness_batch,
                        True)
                    out_image_batch = pyredner.render_pathtracing(
                        scene=out_scene_list, num_samples=128)
                    out_image_batch = out_image_batch.permute(0, 3, 1, 2)
                    out_image_batch = torch.pow(
                        torch.min(
                            out_image_batch,
                            torch.tensor([1.0], device=pyredner.get_device())),
                        1.0 / 2.2)
                    # save_inp_path = os.path.join(
                    #     args.logdir,
                    #     'valid/rotation{}.png'.format(azimuth))
                    # vutils.save_image(out_image_batch, save_inp_path)
                    out_image_batch = out_image_batch.permute(0, 2, 3, 1)
                    output = out_image_batch.cpu().numpy()[0] * 255
                    writer.append_data(output.astype(np.uint8))

                writer.close()
                inp_list.append(inp_image_batch)

            save_inp_list = torch.cat(inp_list, dim=0)
            save_inp_path = os.path.join(
                args.logdir,
                'valid/valid_inp_epoch{}.png'.format(self.cur_epoch, i))
            vutils.save_image(save_inp_list, save_inp_path)

            save_out_list = torch.cat(out_list, dim=0)
            save_out_path = os.path.join(
                args.logdir,
                'valid/valid_out_epoch{}.png'.format(self.cur_epoch, i))
            vutils.save_image(save_out_list, save_out_path)

            self.cur_epoch += 1

            # writer.add_scalar("Surf Loss:", surf_loss.item(), i)
            # writer.add_scalar("Edge Loss:", edge_loss.item(), i)
            # writer.add_scalar("Lap Loss:", lap_loss.item(), i)


trainer = Engine()

for epoch in range(args.epochs):
    trainer.validate()
