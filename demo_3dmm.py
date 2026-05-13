# coding: utf-8

__author__ = 'cleardusk_modified_for_3dmm_txt'

import os
import sys
import argparse
import cv2
import yaml
import numpy as np

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose, calc_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


def save_3dmm_params_txt(param_lst, roi_box_lst, img_fp, out_dir='examples/results'):
    """
    只保存 3DDFA_V2 输出的 3DMM 参数为 txt 文件。
    """

    os.makedirs(out_dir, exist_ok=True)

    old_suffix = get_suffix(img_fp)
    base_name = os.path.basename(img_fp).replace(old_suffix, '')

    param_txt_path = os.path.join(out_dir, f'{base_name}_3dmm_params.txt')
    pose_txt_path = os.path.join(out_dir, f'{base_name}_pose_params.txt')

    # 保存完整 3DMM 参数
    with open(param_txt_path, 'w', encoding='utf-8') as f:
        f.write('3DDFA_V2 3DMM Parameters\n')
        f.write('====================================\n\n')
        f.write('说明：\n')
        f.write('param_lst 是 3DDFA_V2 对每张检测到的人脸回归出的 3DMM 参数。\n')
        f.write('默认模型中，每张脸的参数通常是 62 维：\n')
        f.write('前 12 维：pose / camera 姿态或相机相关参数\n')
        f.write('中间 40 维：shape 身份形状参数\n')
        f.write('最后 10 维：expression 表情参数\n\n')

        f.write(f'Number of faces: {len(param_lst)}\n\n')

        for i, param in enumerate(param_lst):
            param = np.asarray(param)

            f.write(f'Face {i}\n')
            f.write('------------------------------------\n')
            f.write(f'param shape: {param.shape}\n')
            f.write(f'roi box: {roi_box_lst[i]}\n\n')

            f.write('Full 3DMM parameter:\n')
            f.write(np.array2string(param, precision=6, separator=', '))
            f.write('\n\n')

            if param.shape[0] >= 62:
                pose_camera_param = param[:12]
                shape_param = param[12:52]
                expression_param = param[52:62]

                f.write('Pose / Camera parameter, first 12 dims:\n')
                f.write(np.array2string(pose_camera_param, precision=6, separator=', '))
                f.write('\n\n')

                f.write('Shape parameter, 40 dims:\n')
                f.write(np.array2string(shape_param, precision=6, separator=', '))
                f.write('\n\n')

                f.write('Expression parameter, 10 dims:\n')
                f.write(np.array2string(expression_param, precision=6, separator=', '))
                f.write('\n\n')

    # 保存 yaw / pitch / roll 姿态角
    with open(pose_txt_path, 'w', encoding='utf-8') as f:
        f.write('3DDFA_V2 Pose Parameters\n')
        f.write('====================================\n\n')
        f.write('说明：\n')
        f.write('yaw / pitch / roll 是根据 3DMM 参数中的 pose/camera 部分计算出来的人脸姿态角。\n\n')

        for i, param in enumerate(param_lst):
            P, pose = calc_pose(param)

            yaw = pose[0]
            pitch = pose[1]
            roll = pose[2]

            f.write(f'Face {i}\n')
            f.write('------------------------------------\n')
            f.write(f'yaw   : {yaw:.6f}\n')
            f.write(f'pitch : {pitch:.6f}\n')
            f.write(f'roll  : {roll:.6f}\n\n')

            f.write('Camera / pose matrix P:\n')
            f.write(np.array2string(P, precision=6, separator=', '))
            f.write('\n\n')

            print(f'Face {i}: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}')

    print('\nSave txt files:')
    print(f'  {param_txt_path}')
    print(f'  {pose_txt_path}\n')


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    if img is None:
        print(f'Failed to load image: {args.img_fp}')
        sys.exit(-1)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)

    if n == 0:
        print('No face detected, exit')
        sys.exit(-1)

    print(f'Detect {n} faces')

    # 核心：生成 3DMM 参数
    param_lst, roi_box_lst = tddfa(img, boxes)

    # 保存 3DMM 参数 txt
    if args.save_3dmm:
        save_3dmm_params_txt(
            param_lst=param_lst,
            roi_box_lst=roi_box_lst,
            img_fp=args.img_fp,
            out_dir=args.out_dir
        )

    # 如果只想生成 3DMM 参数，到这里结束
    if args.opt == '3dmm':
        return

    # Visualization and serialization
    dense_flag = args.opt in (
        '2d_dense',
        '3d',
        'depth',
        'pncc',
        'uv_tex',
        'ply',
        'obj'
    )

    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    os.makedirs(args.out_dir, exist_ok=True)

    wfp = os.path.join(
        args.out_dir,
        f'{os.path.basename(args.img_fp).replace(old_suffix, "")}_{args.opt}{new_suffix}'
    )

    ver_lst = tddfa.recon_vers(
        param_lst,
        roi_box_lst,
        dense_flag=dense_flag
    )

    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)

    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)

    elif args.opt == '3d':
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)

    elif args.opt == 'depth':
        depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)

    elif args.opt == 'pncc':
        pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)

    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)

    elif args.opt == 'pose':
        viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)

    elif args.opt == 'ply':
        ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)

    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)

    else:
        raise ValueError(f'Unknown opt {args.opt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo of still image of 3DDFA_V2 with 3DMM txt export'
    )

    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')

    parser.add_argument(
        '-o',
        '--opt',
        type=str,
        default='2d_sparse',
        choices=[
            '2d_sparse',
            '2d_dense',
            '3d',
            'depth',
            'pncc',
            'uv_tex',
            'pose',
            'ply',
            'obj',
            '3dmm'
        ],
        help='output option'
    )

    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    parser.add_argument(
        '--save_3dmm',
        type=str2bool,
        default='true',
        help='whether to save 3DMM parameters as txt'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='examples/results',
        help='output directory'
    )

    args = parser.parse_args()
    main(args)

# python demo_3dmm.py \
#   -f examples/inputs/emma.jpg \
#   -o 3dmm \
#   --onnx