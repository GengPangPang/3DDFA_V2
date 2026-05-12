import os

# 必须放在 import pyrender / OpenGL 之前
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import argparse
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

def normalize_mesh(mesh):
    """
    把 mesh 移到原点并缩放到合适大小，方便相机观察。
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geometry for geometry in mesh.geometry.values()]
        )

    vertices = mesh.vertices
    center = vertices.mean(axis=0)
    mesh.apply_translation(-center)

    scale = np.max(mesh.extents)
    if scale > 0:
        mesh.apply_scale(1.8 / scale)

    return mesh


def look_at(eye, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])):
    """
    构造相机位姿矩阵。
    eye: 相机位置
    target: 看向的位置
    up: 相机上方向
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    z_axis = eye - target
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye

    return pose


def add_lights(scene):
    """
    增加多盏灯，让立体感更明显
    """
    light_main = pyrender.DirectionalLight(color=np.ones(3), intensity=3.2)
    light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.8)
    light_rim = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)

    # 主光：右前上方
    pose_main = look_at(eye=[2.5, 2.2, 3.2], up=[0, 1, 0])
    # 补光：左前方
    pose_fill = look_at(eye=[-2.5, 1.0, 2.8], up=[0, 1, 0])
    # 轮廓光：后上方
    pose_rim = look_at(eye=[0.0, 3.0, -3.0], up=[0, 1, 0])

    scene.add(light_main, pose=pose_main)
    scene.add(light_fill, pose=pose_fill)
    scene.add(light_rim, pose=pose_rim)


def get_bg_color(bg_mode):
    if bg_mode == "dark":
        return [40, 44, 52, 255]        # 深灰
    elif bg_mode == "white":
        return [255, 255, 255, 255]     # 纯白
    else:
        return [235, 238, 243, 255]     # 默认浅灰


def get_mesh_color(bg_mode):
    """
    根据背景自动选 mesh 颜色
    """
    if bg_mode == "dark":
        return [0.82, 0.84, 0.88, 1.0]   # 亮灰蓝
    elif bg_mode == "white":
        return [0.45, 0.58, 0.78, 1.0]   # 蓝灰
    else:
        return [0.52, 0.64, 0.82, 1.0]   # 默认蓝灰


def render_one_view(mesh, out_path, eye, up, width=512, height=512, bg_mode="light"):
    bg_color = get_bg_color(bg_mode)
    mesh_color = get_mesh_color(bg_mode)

    scene = pyrender.Scene(
        bg_color=bg_color,
        ambient_light=[0.18, 0.18, 0.18]
    )

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.65,
        baseColorFactor=mesh_color
    )

    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = look_at(eye=eye, up=up)
    scene.add(camera, pose=camera_pose)

    add_lights(scene)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height
    )
    color, depth = renderer.render(scene)
    renderer.delete()

    Image.fromarray(color).save(out_path)
    print("Saved:", out_path)


def make_grid(image_paths, out_path, cols=3):
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    w, h = imgs[0].size

    font_size = 36
    title_h = font_size + 20

    rows = (len(imgs) + cols - 1) // cols

    canvas = Image.new("RGB", (cols * w, rows * (h + title_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        font_size
    )

    for i, (img, path) in enumerate(zip(imgs, image_paths)):
        row = i // cols
        col = i % cols

        x = col * w
        y = row * (h + title_h)

        canvas.paste(img, (x, y))
        title = os.path.splitext(os.path.basename(path))[0]
        draw.text((x + 10, y + h + 8), title, fill=(0, 0, 0), font=font)

    canvas.save(out_path)
    print("Saved:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True, help="Path to OBJ file")
    parser.add_argument("--out_dir", default="outputs/opengl_renders")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument(
        "--bg_mode",
        choices=["light", "dark", "white"],
        default="light",
        help="Background mode"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mesh = trimesh.load(args.obj, force="mesh")
    mesh = normalize_mesh(mesh)

    views = {
        "front": {
            "eye": [0.0, 0.0, 2.8],
            "up": [0.0, 1.0, 0.0],
        },
        "left": {
            "eye": [-2.8, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
        },
        "right": {
            "eye": [2.8, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
        },
        "top": {
            "eye": [0.0, 2.8, 0.01],
            "up": [0.0, 0.0, -1.0],
        },
        "iso": {
            "eye": [1.8, 1.5, 2.2],
            "up": [0.0, 1.0, 0.0],
        },
    }

    saved = []
    for name, cfg in views.items():
        out_path = os.path.join(args.out_dir, f"{name}.png")
        render_one_view(
            mesh=mesh,
            out_path=out_path,
            eye=cfg["eye"],
            up=cfg["up"],
            width=args.size,
            height=args.size,
            bg_mode=args.bg_mode,
        )
        saved.append(out_path)

    make_grid(saved, os.path.join(args.out_dir, "multi_view_grid.png"), cols=3)


if __name__ == "__main__":
    main()

# python render_opengl_obj.py \
#   --obj examples/results/emma_obj.obj \
#   --out_dir outputs/opengl_renders \
#   --bg_mode light