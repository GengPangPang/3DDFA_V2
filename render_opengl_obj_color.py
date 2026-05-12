# coding: utf-8
import os

# 必须放在 import pyrender / OpenGL 之前
# osmesa 适合无图形界面的 WSL / Linux 离屏渲染
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import argparse
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont


def normalize_mesh(mesh):
    """
    将 mesh 平移到原点附近，并缩放到统一大小，方便相机观察。
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


def look_at(
    eye,
    target=np.array([0.0, 0.0, 0.0]),
    up=np.array([0.0, 1.0, 0.0])
):
    """
    构造相机位姿矩阵。
    eye: 相机位置
    target: 相机看向的位置
    up: 相机上方向
    """
    eye = np.array(eye, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    z_axis = eye - target
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        raise ValueError("Camera eye and target are too close.")
    z_axis = z_axis / z_norm

    x_axis = np.cross(up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        raise ValueError("Camera up vector is parallel to view direction.")
    x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)

    pose = np.eye(4)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye

    return pose


def add_lights(scene):
    """
    增加多盏灯，使模型的立体感更明显。
    """
    light_main = pyrender.DirectionalLight(color=np.ones(3), intensity=2.8)
    light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.4)
    light_rim = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)

    pose_main = look_at(eye=[2.5, 2.2, 3.2], up=[0, 1, 0])
    pose_fill = look_at(eye=[-2.5, 1.0, 2.8], up=[0, 1, 0])
    pose_rim = look_at(eye=[0.0, 3.0, -3.0], up=[0, 1, 0])

    scene.add(light_main, pose=pose_main)
    scene.add(light_fill, pose=pose_fill)
    scene.add(light_rim, pose=pose_rim)


def get_bg_color(bg_mode):
    if bg_mode == "dark":
        return [40, 44, 52, 255]
    if bg_mode == "white":
        return [255, 255, 255, 255]
    return [235, 238, 243, 255]


def get_material_color(bg_mode):
    """
    统一材质模式下使用的 mesh 颜色。
    """
    if bg_mode == "dark":
        return [0.82, 0.84, 0.88, 1.0]
    if bg_mode == "white":
        return [0.45, 0.58, 0.78, 1.0]
    return [0.52, 0.64, 0.82, 1.0]


def mesh_has_vertex_color(mesh):
    """
    判断 trimesh 是否读到了顶点颜色。
    3DDFA_V2 导出的 OBJ 可能是:
        v x y z r g b
    如果 trimesh 成功读取，通常会存在 mesh.visual.vertex_colors。
    """
    if not hasattr(mesh, "visual"):
        return False

    if not hasattr(mesh.visual, "vertex_colors"):
        return False

    colors = mesh.visual.vertex_colors
    if colors is None:
        return False

    colors = np.asarray(colors)

    if colors.ndim != 2:
        return False

    if colors.shape[0] != len(mesh.vertices):
        return False

    if colors.shape[1] < 3:
        return False

    # 如果所有颜色几乎一样，说明可能没有真正读到有效颜色
    rgb = colors[:, :3].astype(np.float32)
    if np.max(rgb) - np.min(rgb) < 1e-6:
        return False

    return True


def ensure_vertex_colors_uint8(mesh):
    """
    确保顶点颜色是 RGBA uint8 格式。
    有些 OBJ 里的颜色是 0~1 浮点数，有些是 0~255。
    """
    if not mesh_has_vertex_color(mesh):
        return mesh

    colors = np.asarray(mesh.visual.vertex_colors)

    if colors.shape[1] == 3:
        alpha = np.full((colors.shape[0], 1), 255, dtype=colors.dtype)
        colors = np.concatenate([colors, alpha], axis=1)

    colors = colors[:, :4].astype(np.float32)

    # 如果是 0~1 浮点颜色，转成 0~255
    if colors[:, :3].max() <= 1.0:
        colors[:, :3] *= 255.0

    colors[:, 3] = 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)

    return mesh


def create_pyrender_mesh(mesh, color_mode, bg_mode):
    """
    根据 color_mode 创建 pyrender mesh。

    color_mode:
        auto     : 有顶点颜色则使用顶点颜色，否则使用统一材质
        vertex   : 强制使用 OBJ 顶点颜色
        material : 强制使用统一材质
    """
    has_color = mesh_has_vertex_color(mesh)

    if color_mode == "vertex":
        if not has_color:
            print("[Warning] OBJ does not contain valid vertex colors. Falling back to material color.")
        else:
            mesh = ensure_vertex_colors_uint8(mesh)
            print("[Info] Using OBJ vertex colors.")
            return pyrender.Mesh.from_trimesh(mesh, smooth=True)

    if color_mode == "auto" and has_color:
        mesh = ensure_vertex_colors_uint8(mesh)
        print("[Info] Vertex colors detected. Using OBJ vertex colors.")
        return pyrender.Mesh.from_trimesh(mesh, smooth=True)

    material_color = get_material_color(bg_mode)
    print("[Info] Using uniform material color:", material_color)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.65,
        baseColorFactor=material_color
    )

    return pyrender.Mesh.from_trimesh(
        mesh,
        material=material,
        smooth=True
    )


def render_one_view(
    mesh,
    out_path,
    eye,
    up,
    width=512,
    height=512,
    bg_mode="light",
    color_mode="auto"
):
    bg_color = get_bg_color(bg_mode)

    scene = pyrender.Scene(
        bg_color=bg_color,
        ambient_light=[0.22, 0.22, 0.22]
    )

    render_mesh = create_pyrender_mesh(
        mesh=mesh,
        color_mode=color_mode,
        bg_mode=bg_mode
    )
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


def make_grid(image_paths, out_path, cols=3, font_size=36):
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    w, h = imgs[0].size

    title_h = font_size + 20
    rows = (len(imgs) + cols - 1) // cols

    canvas = Image.new(
        "RGB",
        (cols * w, rows * (h + title_h)),
        (255, 255, 255)
    )
    draw = ImageDraw.Draw(canvas)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

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


def print_mesh_info(mesh):
    print("========== Mesh Info ==========")
    print("vertices:", len(mesh.vertices))
    print("faces:", len(mesh.faces))
    print("has vertex colors:", mesh_has_vertex_color(mesh))

    if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
        colors = mesh.visual.vertex_colors
        if colors is not None:
            colors = np.asarray(colors)
            print("vertex color shape:", colors.shape)
            if colors.size > 0:
                print("vertex color min:", colors[:, :3].min())
                print("vertex color max:", colors[:, :3].max())

    print("================================")


def main():
    parser = argparse.ArgumentParser(
        description="Render 3DDFA_V2 OBJ with vertex colors or uniform material."
    )
    parser.add_argument(
        "--obj",
        required=True,
        help="Path to OBJ file exported by 3DDFA_V2"
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/opengl_renders_color",
        help="Output directory"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image size"
    )
    parser.add_argument(
        "--bg_mode",
        choices=["light", "dark", "white"],
        default="light",
        help="Background mode"
    )
    parser.add_argument(
        "--color_mode",
        choices=["auto", "vertex", "material"],
        default="auto",
        help="auto: use vertex colors if available; vertex: force vertex colors; material: use uniform material"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mesh = trimesh.load(args.obj, force="mesh")
    mesh = normalize_mesh(mesh)

    print_mesh_info(mesh)

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
            color_mode=args.color_mode
        )

        saved.append(out_path)

    grid_path = os.path.join(args.out_dir, "multi_view_grid.png")
    make_grid(saved, grid_path, cols=3)


if __name__ == "__main__":
    main()

