import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional
from torchvision.utils import make_grid



#### render results
def render_on_img(fx, fy, cx, cy, input_img, body_trimesh, material, light, camera_pose, renderer):
    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx, fy=fy,
        cx=cx, cy=cy)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)
    scene.add(body_mesh, 'body_mesh')
    color, _ = renderer.render(scene)
    color = color[:, :, ::-1]
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    render_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * input_img)
    render_img = np.ascontiguousarray(render_img, dtype=np.uint8)
    # cv2.imwrite(os.path.join(output_img_save_folder, '{}_{}'.format(recording_name, frame_name)), pred_render_img)
    return render_img


def render_in_scene(fx, fy, cx, cy, body_trimesh, scene_trimesh, material, light, camera_pose, renderer):
    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx, fy=fy,
        cx=cx, cy=cy)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)
    static_scene_mesh = pyrender.Mesh.from_trimesh(scene_trimesh)
    scene = pyrender.Scene()
    scene.add(camera, pose=camera_pose)
    scene.add(light, pose=camera_pose)
    scene.add(static_scene_mesh, 'scene_mesh')
    scene.add(body_mesh, 'body_mesh')
    render_img, _ = renderer.render(scene)
    render_img = render_img[:, :, ::-1]
    render_img = np.ascontiguousarray(render_img)
    return render_img
