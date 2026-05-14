# coding: utf-8
"""
Unified 3DDFA_V2 pipeline:
- detect ALL faces in one image
- save per-face params / pose / geometry / visualizations
- frontalize each detected face using 3DDFA_V2 pose
- render frontalized faces with OpenGL/pyrender
- create a grid of all frontalized face renders

Run from the root directory of 3DDFA_V2, for example:

python unified_face_pipeline_all_frontal.py \
  -f examples/inputs/emma.jpg \
  --out_dir outputs/emma_all_faces \
  --onnx \
  --render_frontal \
  --bg_mode light

Key outputs:
  out_dir/
    manifest.json
    params/
      detected_boxes_all.npy/json
      raw_params_all.npy/json
      roi_boxes_all.npy/json
      selected_face_indices.npy/json
    face_000/
      params/
      pose/
      geometry/
      visualizations/
      renders_frontal/
        front.png
        left.png
        right.png
        iso.png
        frontal_views_grid.png
    face_001/
      ...
    frontalized_all_faces_grid.png
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import yaml
import numpy as np

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render  # Optional faster renderer
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks
from utils.tddfa_util import str2bool


ALL_OPTS = ("2d_sparse", "2d_dense", "3d", "depth", "pncc", "uv_tex", "pose", "ply", "obj")
DENSE_OPTS = {"2d_dense", "3d", "depth", "pncc", "uv_tex", "ply", "obj"}
VISUAL_OPTS = {"2d_sparse", "2d_dense", "3d", "depth", "pncc", "uv_tex", "pose"}
GEOMETRY_OPTS = {"ply", "obj"}


# ----------------------------- Basic utils -----------------------------

def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return to_builtin(x.tolist())
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    return x


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def save_array_json_pair(stem: Path, arr: Any) -> None:
    np.save(str(stem.with_suffix(".npy")), arr, allow_pickle=True)
    save_json(stem.with_suffix(".json"), arr)


def split_param(param: np.ndarray) -> Dict[str, Any]:
    """
    3DDFA_V2 common layout:
      first 12: pose / projection matrix P flattened as 3x4
      next 40: identity shape coeffs
      last 10: expression coeffs
    """
    p = np.asarray(param).reshape(-1)
    out: Dict[str, Any] = {
        "raw_param": p,
        "length": int(p.shape[0]),
    }
    if p.shape[0] >= 12:
        out["pose_12"] = p[:12]
        out["pose_matrix_3x4"] = p[:12].reshape(3, 4)
    if p.shape[0] >= 52:
        out["shape_coeff_40"] = p[12:52]
    if p.shape[0] >= 62:
        out["expression_coeff_10"] = p[52:62]
    if p.shape[0] > 62:
        out["extra_coeff"] = p[62:]
    return out


def try_extract_pose_angles(param: np.ndarray) -> Dict[str, Any]:
    """
    Extract scale / R / t / pitch-yaw-roll from 3DDFA_V2 pose.
    """
    result: Dict[str, Any] = {}
    try:
        from utils.pose import P2sRt, matrix2angle  # type: ignore
        P = np.asarray(param).reshape(-1)[:12].reshape(3, 4)
        s, R, t3d = P2sRt(P)
        pitch, yaw, roll = matrix2angle(R)
        result.update({
            "scale": s,
            "rotation_matrix": R,
            "translation_3d": t3d,
            "pitch_yaw_roll": [pitch, yaw, roll],
        })
    except Exception as e:
        result["pose_helper_warning"] = f"Could not extract scale/R/t/angles: {e}"
    return result


def init_models(config: str, mode: str, onnx: bool):
    cfg = yaml.load(open(config), Loader=yaml.SafeLoader)
    if onnx:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
        os.environ["OMP_NUM_THREADS"] = "4"
        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX
        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = mode == "gpu"
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()
    return face_boxes, tddfa


# ----------------------- Existing 3DDFA outputs -----------------------

def write_one_opt(
    opt: str,
    img_bgr: np.ndarray,
    ver_lst: Sequence[np.ndarray],
    param_lst: Sequence[np.ndarray],
    tddfa: Any,
    out_path: Path,
    show_flag: bool = False,
) -> None:
    mkdir(out_path.parent)

    if opt in ("2d_sparse", "2d_dense"):
        draw_landmarks(
            img_bgr.copy(),
            ver_lst,
            show_flag=show_flag,
            dense_flag=(opt == "2d_dense"),
            wfp=str(out_path),
        )
    elif opt == "3d":
        render(img_bgr.copy(), ver_lst, tddfa.tri, alpha=0.6, show_flag=show_flag, wfp=str(out_path))
    elif opt == "depth":
        depth(img_bgr.copy(), ver_lst, tddfa.tri, show_flag=show_flag, wfp=str(out_path), with_bg_flag=True)
    elif opt == "pncc":
        pncc(img_bgr.copy(), ver_lst, tddfa.tri, show_flag=show_flag, wfp=str(out_path), with_bg_flag=True)
    elif opt == "uv_tex":
        uv_tex(img_bgr.copy(), ver_lst, tddfa.tri, show_flag=show_flag, wfp=str(out_path))
    elif opt == "pose":
        viz_pose(img_bgr.copy(), param_lst, ver_lst, show_flag=show_flag, wfp=str(out_path))
    elif opt == "ply":
        ser_to_ply(ver_lst, tddfa.tri, height=img_bgr.shape[0], wfp=str(out_path))
    elif opt == "obj":
        ser_to_obj(img_bgr.copy(), ver_lst, tddfa.tri, height=img_bgr.shape[0], wfp=str(out_path))
    else:
        raise ValueError(f"Unknown opt: {opt}")


# ---------------------- Mesh / frontalization utils ----------------------

def ver_to_vertices(ver: np.ndarray) -> np.ndarray:
    """
    Convert 3DDFA vertices to (N, 3).
    Usually ver is (3, N).
    """
    v = np.asarray(ver)
    if v.ndim != 2:
        raise ValueError(f"Unexpected vertex shape: {v.shape}")
    if v.shape[0] == 3 and v.shape[1] != 3:
        return v.T.astype(np.float64)
    if v.shape[1] == 3:
        return v.astype(np.float64)
    raise ValueError(f"Cannot convert ver to (N, 3), got shape {v.shape}")


def tri_to_faces(tri: np.ndarray) -> np.ndarray:
    """
    Convert tddfa.tri to (F, 3).
    """
    t = np.asarray(tri)
    if t.ndim != 2:
        raise ValueError(f"Unexpected tri shape: {t.shape}")
    if t.shape[0] == 3 and t.shape[1] != 3:
        return t.T.astype(np.int64)
    if t.shape[1] == 3:
        return t.astype(np.int64)
    raise ValueError(f"Cannot convert tri to (F, 3), got shape {t.shape}")


def frontalize_vertices(vertices: np.ndarray, R: np.ndarray, frontal_rot: str = "R") -> np.ndarray:
    """
    Frontalize face by applying inverse pose to the reconstructed mesh.

    vertices: (N, 3)
    R: (3, 3) rotation matrix from 3DDFA_V2
    frontal_rot:
      - 'R'  : V' = (V - c) @ R
      - 'RT' : V' = (V - c) @ R.T

    Because different forks / export conventions may differ, if frontal direction
    looks wrong, switch R <-> RT.
    """
    V = np.asarray(vertices, dtype=np.float64)
    center = V.mean(axis=0, keepdims=True)
    V0 = V - center

    if frontal_rot == "R":
        V1 = V0 @ R
    elif frontal_rot == "RT":
        V1 = V0 @ R.T
    else:
        raise ValueError(f"Unsupported frontal_rot: {frontal_rot}")

    return V1


def normalize_vertices(vertices: np.ndarray, target_extent: float = 1.8) -> np.ndarray:
    V = np.asarray(vertices, dtype=np.float64)
    center = V.mean(axis=0, keepdims=True)
    V = V - center
    extent = np.max(np.max(V, axis=0) - np.min(V, axis=0))
    if extent > 0:
        V = V * (target_extent / extent)
    return V


def center_and_scale_vertices(
    vertices: np.ndarray,
    scale_mode: str = "fixed",
    fixed_render_scale: float = 120.0,
    target_extent: float = 1.8,
) -> np.ndarray:
    """
    Prepare vertices for OpenGL rendering.

    scale_mode:
      - fixed: only center, then divide by a fixed pixel scale. This preserves
        absolute geometric differences better across faces in the same image.
      - per_face: old behavior; center each face and scale it to target_extent.
        This is convenient for display but makes different people look more alike.
    """
    V = np.asarray(vertices, dtype=np.float64)
    V = V - V.mean(axis=0, keepdims=True)

    if scale_mode == "fixed":
        if fixed_render_scale <= 0:
            raise ValueError("fixed_render_scale must be > 0")
        return V / float(fixed_render_scale)

    if scale_mode == "per_face":
        extent = np.max(np.max(V, axis=0) - np.min(V, axis=0))
        if extent > 0:
            V = V * (target_extent / extent)
        return V

    raise ValueError(f"Unsupported scale_mode: {scale_mode}")


def sample_vertex_colors_from_image(
    img_bgr: np.ndarray,
    vertices_image_space: np.ndarray,
    default_rgba=(0.72, 0.72, 0.72, 1.0),
) -> np.ndarray:
    """
    Approximate texture by sampling the original image at each 3DDFA vertex.

    3DDFA vertices are in image coordinates before frontalization, so sample colors
    BEFORE rotating/centering/scaling the mesh. This keeps identity cues such as
    eyebrows, lips, skin tone, beard, wrinkles, and local shading.
    """
    H, W = img_bgr.shape[:2]
    V = np.asarray(vertices_image_space, dtype=np.float64)
    xy = np.rint(V[:, :2]).astype(np.int64)

    colors = np.empty((V.shape[0], 4), dtype=np.float32)
    colors[:] = np.asarray(default_rgba, dtype=np.float32)

    valid = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
    if np.any(valid):
        bgr = img_bgr[xy[valid, 1], xy[valid, 0]].astype(np.float32) / 255.0
        rgb = bgr[:, ::-1]
        colors[valid, :3] = rgb
        colors[valid, 3] = 1.0

    return colors


# ------------------------ OpenGL render helpers ------------------------

def look_at(eye, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0])):
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


def get_bg_color(bg_mode: str):
    if bg_mode == "dark":
        return [40, 44, 52, 255]
    if bg_mode == "white":
        return [255, 255, 255, 255]
    return [235, 238, 243, 255]


def get_mesh_color(bg_mode: str):
    if bg_mode == "dark":
        return [0.82, 0.84, 0.88, 1.0]
    if bg_mode == "white":
        return [0.45, 0.58, 0.78, 1.0]
    return [0.52, 0.64, 0.82, 1.0]


def add_lights(scene):
    import pyrender

    light_main = pyrender.DirectionalLight(color=np.ones(3), intensity=3.2)
    light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.8)
    light_rim = pyrender.DirectionalLight(color=np.ones(3), intensity=1.2)

    scene.add(light_main, pose=look_at(eye=[2.5, 2.2, 3.2], up=[0, 1, 0]))
    scene.add(light_fill, pose=look_at(eye=[-2.5, 1.0, 2.8], up=[0, 1, 0]))
    scene.add(light_rim, pose=look_at(eye=[0.0, 3.0, -3.0], up=[0, 1, 0]))


def render_vertices_one_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    eye,
    up,
    width=512,
    height=512,
    bg_mode="light",
    vertex_colors: Optional[np.ndarray] = None,
    render_color_mode: str = "texture",
):
    import trimesh
    import pyrender
    from PIL import Image

    mesh_tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # If the mesh was mirrored between image coords and OpenGL coords, normals
    # can become inconsistent. This makes lighting stable.
    try:
        mesh_tm.fix_normals()
    except Exception:
        pass

    scene = pyrender.Scene(
        bg_color=get_bg_color(bg_mode),
        ambient_light=[0.35, 0.35, 0.35]
    )

    if render_color_mode == "texture" and vertex_colors is not None:
        mesh_tm.visual.vertex_colors = np.asarray(vertex_colors)
        render_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
    else:
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.65,
            baseColorFactor=get_mesh_color(bg_mode),
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh_tm, material=material, smooth=True)

    scene.add(render_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=look_at(eye=eye, up=up))

    add_lights(scene)

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _ = renderer.render(scene)
    renderer.delete()

    mkdir(out_path.parent)
    Image.fromarray(color).save(out_path)


def make_grid(image_paths: Sequence[Path], out_path: Path, cols=3):
    from PIL import Image, ImageDraw, ImageFont

    if len(image_paths) == 0:
        return

    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    w, h = imgs[0].size
    font_size = 28
    title_h = font_size + 18

    rows = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * (h + title_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, path) in enumerate(zip(imgs, image_paths)):
        row = i // cols
        col = i % cols
        x = col * w
        y = row * (h + title_h)
        canvas.paste(img, (x, y))
        draw.text((x + 10, y + h + 5), path.parent.parent.name, fill=(0, 0, 0), font=font)

    mkdir(out_path.parent)
    canvas.save(out_path)


def render_frontal_views_for_one_face(
    ver_dense: np.ndarray,
    tri: np.ndarray,
    R: Optional[np.ndarray],
    out_dir: Path,
    img_bgr: Optional[np.ndarray] = None,
    render_size: int = 512,
    bg_mode: str = "light",
    frontal_rot: str = "R",
    render_color_mode: str = "texture",
    scale_mode: str = "fixed",
    fixed_render_scale: float = 120.0,
    flip_y_for_opengl: bool = True,
) -> Dict[str, str]:
    """
    Render frontalized views for one face while preserving identity cues.

    Key differences from the old version:
      1) Samples vertex colors from the original image before frontalization.
      2) Uses fixed scaling by default instead of per-face normalization.
      3) Corrects image-coordinate y-down to OpenGL y-up, and reverses face
         winding so the mesh does not become transparent after mirroring.
    """
    mkdir(out_dir)

    vertices_img = ver_to_vertices(ver_dense)
    faces = tri_to_faces(tri)

    vertex_colors = None
    if img_bgr is not None and render_color_mode == "texture":
        vertex_colors = sample_vertex_colors_from_image(img_bgr, vertices_img)

    vertices = vertices_img.copy()

    if R is not None:
        vertices = frontalize_vertices(vertices, np.asarray(R, dtype=np.float64), frontal_rot=frontal_rot)
    else:
        vertices = vertices - vertices.mean(axis=0, keepdims=True)

    vertices = center_and_scale_vertices(
        vertices,
        scale_mode=scale_mode,
        fixed_render_scale=fixed_render_scale,
        target_extent=1.8,
    )

    if flip_y_for_opengl:
        vertices[:, 1] *= -1
        # Single-axis mirroring reverses handedness. Reverse triangle winding
        # to keep normals/front faces correct and avoid transparent-looking mesh.
        faces = faces[:, [0, 2, 1]]

    views = {
        "front": {"eye": [0.0, 0.0, 2.8], "up": [0.0, 1.0, 0.0]},
        "left":  {"eye": [-2.8, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
        "right": {"eye": [2.8, 0.0, 0.0], "up": [0.0, 1.0, 0.0]},
        "iso":   {"eye": [1.8, 1.5, 2.2], "up": [0.0, 1.0, 0.0]},
    }

    saved = {}
    saved_paths = []

    for name, cfg in views.items():
        out_path = out_dir / f"{name}.png"
        render_vertices_one_view(
            vertices=vertices,
            faces=faces,
            out_path=out_path,
            eye=cfg["eye"],
            up=cfg["up"],
            width=render_size,
            height=render_size,
            bg_mode=bg_mode,
            vertex_colors=vertex_colors,
            render_color_mode=render_color_mode,
        )
        saved[name] = str(out_path)
        saved_paths.append(out_path)

    grid_path = out_dir / "frontal_views_grid.png"
    make_grid(saved_paths, grid_path, cols=2)
    saved["grid"] = str(grid_path)

    return saved


# ----------------------------- Face selection -----------------------------

def select_faces(
    boxes: Sequence[Any],
    param_lst: Sequence[np.ndarray],
    roi_box_lst: Sequence[np.ndarray],
    face_index: int,
) -> Tuple[List[Any], List[np.ndarray], List[np.ndarray], List[int]]:
    if face_index < 0:
        indices = list(range(len(boxes)))
    else:
        if face_index >= len(boxes):
            raise IndexError(f"--face_index {face_index} out of range; detected {len(boxes)} face(s).")
        indices = [face_index]

    return (
        [boxes[i] for i in indices],
        [param_lst[i] for i in indices],
        [roi_box_lst[i] for i in indices],
        indices,
    )


# ------------------------------- Main pipeline -------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    img_path = Path(args.img_fp)
    out_dir = Path(args.out_dir)
    mkdir(out_dir)

    global_params_dir = out_dir / "params"
    mkdir(global_params_dir)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    face_boxes, tddfa = init_models(args.config, args.mode, args.onnx)

    boxes = face_boxes(img)
    if len(boxes) == 0:
        raise RuntimeError("No face detected.")

    print(f"Detected {len(boxes)} face(s).")

    param_lst_all, roi_box_lst_all = tddfa(img, boxes)

    _, param_lst, roi_box_lst, indices = select_faces(
        boxes, param_lst_all, roi_box_lst_all, args.face_index
    )

    # Save global metadata
    save_array_json_pair(global_params_dir / "detected_boxes_all", np.asarray(boxes, dtype=object))
    save_array_json_pair(global_params_dir / "selected_face_indices", np.asarray(indices))
    save_array_json_pair(global_params_dir / "raw_params_all", np.asarray(param_lst_all, dtype=object))
    save_array_json_pair(global_params_dir / "roi_boxes_all", np.asarray(roi_box_lst_all, dtype=object))

    # Reconstruct all faces once
    ver_sparse_all = tddfa.recon_vers(param_lst_all, roi_box_lst_all, dense_flag=False)
    ver_dense_all = tddfa.recon_vers(param_lst_all, roi_box_lst_all, dense_flag=True)

    front_render_paths: List[Path] = []
    face_manifests: List[Dict[str, Any]] = []

    for local_i, global_face_idx in enumerate(indices):
        print(f"\nProcessing face index: {global_face_idx}")

        face_dir = out_dir / f"face_{global_face_idx:03d}"
        params_dir = face_dir / "params"
        pose_dir = face_dir / "pose"
        vis_dir = face_dir / "visualizations"
        geom_dir = face_dir / "geometry"
        renders_frontal_dir = face_dir / "renders_frontal"

        for d in (params_dir, pose_dir, vis_dir, geom_dir, renders_frontal_dir):
            mkdir(d)

        param = np.asarray(param_lst_all[global_face_idx])
        roi_box = np.asarray(roi_box_lst_all[global_face_idx])
        ver_sparse = ver_sparse_all[global_face_idx]
        ver_dense = ver_dense_all[global_face_idx]

        # Save params
        save_array_json_pair(params_dir / "raw_param", param)
        save_array_json_pair(params_dir / "roi_box", roi_box)
        np.save(str(params_dir / "vertices_sparse.npy"), np.asarray(ver_sparse), allow_pickle=True)
        np.save(str(params_dir / "vertices_dense.npy"), np.asarray(ver_dense), allow_pickle=True)

        split_info = split_param(param)
        save_json(params_dir / "raw_param_split.json", split_info)

        # Save pose
        pose_info = split_param(param)
        pose_info.update(try_extract_pose_angles(param))
        pose_info["selected_face_index"] = global_face_idx
        save_json(pose_dir / "pose_params.json", pose_info)
        np.save(str(pose_dir / "pose_params.npy"), np.asarray(pose_info, dtype=object), allow_pickle=True)

        if param.reshape(-1).shape[0] >= 12:
            pose_mat = param.reshape(-1)[:12].reshape(3, 4)
            save_array_json_pair(pose_dir / "pose_matrix_3x4", pose_mat)

        # Standard per-face 3DDFA outputs
        written_files = {}

        if args.write_standard_outputs:
            for opt in args.opts:
                if opt in VISUAL_OPTS:
                    out_path = vis_dir / f"{opt}.jpg"
                elif opt in GEOMETRY_OPTS:
                    out_path = geom_dir / f"face.{opt}"
                else:
                    raise ValueError(f"Unsupported opt: {opt}")

                ver_lst = [ver_dense] if opt in DENSE_OPTS else [ver_sparse]
                write_one_opt(opt, img, ver_lst, [param], tddfa, out_path, show_flag=args.show_flag)
                written_files[opt] = str(out_path)
                print(f"Saved {opt}: {out_path}")

        # Frontalized render
        frontal_written = {}
        if args.render_frontal:
            R = pose_info.get("rotation_matrix", None)
            frontal_written = render_frontal_views_for_one_face(
                ver_dense=ver_dense,
                tri=tddfa.tri,
                R=R,
                out_dir=renders_frontal_dir,
                img_bgr=img,
                render_size=args.render_size,
                bg_mode=args.bg_mode,
                frontal_rot=args.frontal_rot,
                render_color_mode=args.render_color_mode,
                scale_mode=args.scale_mode,
                fixed_render_scale=args.fixed_render_scale,
                flip_y_for_opengl=args.flip_y_for_opengl,
            )
            if "front" in frontal_written:
                front_render_paths.append(Path(frontal_written["front"]))
            print(f"Saved frontal renders: {renders_frontal_dir}")

        one_manifest = {
            "face_index": global_face_idx,
            "roi_box": to_builtin(roi_box),
            "pose_pitch_yaw_roll": to_builtin(pose_info.get("pitch_yaw_roll", None)),
            "standard_outputs": written_files,
            "frontal_outputs": frontal_written,
        }
        save_json(face_dir / "manifest.json", one_manifest)
        face_manifests.append(one_manifest)

    # Grid for all frontal faces
    all_faces_grid = None
    if args.render_frontal and len(front_render_paths) > 0:
        all_faces_grid = out_dir / "frontalized_all_faces_grid.png"
        make_grid(front_render_paths, all_faces_grid, cols=args.grid_cols)
        print(f"Saved all-face frontal grid: {all_faces_grid}")

    manifest = {
        "input_image": str(img_path),
        "output_dir": str(out_dir),
        "detected_faces": len(boxes),
        "selected_face_indices": indices,
        "frontal_rot": args.frontal_rot,
        "render_color_mode": args.render_color_mode,
        "scale_mode": args.scale_mode,
        "fixed_render_scale": args.fixed_render_scale,
        "flip_y_for_opengl": args.flip_y_for_opengl,
        "all_faces_frontal_grid": str(all_faces_grid) if all_faces_grid is not None else None,
        "faces": face_manifests,
    }
    save_json(out_dir / "manifest.json", manifest)
    print(f"\nDone. Manifest: {out_dir / 'manifest.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3DDFA_V2 all-face frontalization + per-face frontal rendering pipeline"
    )
    parser.add_argument("-c", "--config", type=str, default="configs/mb1_120x120.yml")
    parser.add_argument("-f", "--img_fp", type=str, required=True, help="Input face image path")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("-m", "--mode", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--onnx", action="store_true", default=False)
    parser.add_argument("--show_flag", type=str2bool, default=False)

    parser.add_argument(
        "--opts",
        nargs="+",
        default=list(ALL_OPTS),
        choices=list(ALL_OPTS),
        help="3DDFA standard outputs to generate per face. Default: all.",
    )
    parser.add_argument(
        "--face_index",
        type=int,
        default=-1,
        help="Which detected face to export. -1 means ALL detected faces.",
    )

    parser.add_argument(
        "--write_standard_outputs",
        type=str2bool,
        default=True,
        help="Whether to also write the usual 3DDFA outputs per face.",
    )

    parser.add_argument(
        "--render_frontal",
        type=str2bool,
        default=True,
        help="Whether to render frontalized views per detected face.",
    )
    parser.add_argument("--render_size", type=int, default=512)
    parser.add_argument("--bg_mode", choices=["light", "dark", "white"], default="light")
    parser.add_argument(
        "--render_color_mode",
        choices=["texture", "solid"],
        default="texture",
        help="texture samples original image colors at mesh vertices; solid uses one material color.",
    )
    parser.add_argument(
        "--scale_mode",
        choices=["fixed", "per_face"],
        default="fixed",
        help="fixed preserves relative size/shape cues better; per_face is the old normalize-to-same-extent behavior.",
    )
    parser.add_argument(
        "--fixed_render_scale",
        type=float,
        default=120.0,
        help="Pixel-to-render scale used when --scale_mode fixed. Try 80, 120, 160 depending on face size.",
    )
    parser.add_argument(
        "--flip_y_for_opengl",
        type=str2bool,
        default=True,
        help="Flip y from image coordinates to OpenGL coordinates and reverse face winding.",
    )
    parser.add_argument(
        "--frontal_rot",
        choices=["R", "RT"],
        default="R",
        help=(
            "How to apply 3DDFA rotation during frontalization. "
            "If the frontal face looks wrong, try switching from R to RT."
        ),
    )
    parser.add_argument("--grid_cols", type=int, default=3)

    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())

# python unified_face_pipeline_identity_preserving_combined_views.py \
#   -f examples/inputs/1.png \
#   --out_dir outputs/1_identity_preserving_combined \
#   --onnx \
#   --render_frontal true \
#   --render_combined_views true \
#   --render_color_mode texture \
#   --scale_mode fixed \
#   --fixed_render_scale 120 \
#   --combined_layout row \
#   --bg_mode light