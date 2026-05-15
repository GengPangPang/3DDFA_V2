from pathlib import Path


def check_obj(obj_path):
    obj_path = Path(obj_path)

    num_vertices = 0
    num_faces = 0

    min_face_index = float("inf")
    max_face_index = 0

    non_triangle_faces = 0
    invalid_face_indices = 0
    invalid_vertex_lines = 0
    invalid_face_lines = 0

    has_vertex_color = False
    vertex_color_count = 0

    with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # 顶点行：v x y z 或 v x y z r g b
            if parts[0] == "v":
                num_vertices += 1

                if len(parts) == 7:
                    has_vertex_color = True
                    vertex_color_count += 1
                elif len(parts) == 4:
                    pass
                else:
                    invalid_vertex_lines += 1
                    print(f"[Invalid vertex line] line {line_no}: {line}")

                # 检查顶点坐标和颜色是否能转成 float
                try:
                    values = [float(x) for x in parts[1:]]
                except ValueError:
                    invalid_vertex_lines += 1
                    print(f"[Invalid vertex value] line {line_no}: {line}")

                # 如果有颜色，检查 RGB 是否在 0~1 之间
                if len(parts) == 7:
                    r, g, b = map(float, parts[4:7])
                    if not (0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0):
                        print(f"[Warning] vertex color out of range at line {line_no}: {r}, {g}, {b}")

            # 面片行：f i j k
            elif parts[0] == "f":
                num_faces += 1
                face_tokens = parts[1:]

                if len(face_tokens) != 3:
                    non_triangle_faces += 1
                    print(f"[Non-triangle face] line {line_no}: {line}")

                for token in face_tokens:
                    try:
                        # 兼容 f 1 2 3、f 1/1 2/2 3/3、f 1/1/1 2/2/2 3/3/3
                        idx = int(token.split("/")[0])
                    except ValueError:
                        invalid_face_lines += 1
                        print(f"[Invalid face index] line {line_no}: {line}")
                        continue

                    min_face_index = min(min_face_index, idx)
                    max_face_index = max(max_face_index, idx)

            # 其他 OBJ 行，比如 vt、vn、mtllib、usemtl，这里不作为错误
            else:
                pass

    if num_faces == 0:
        min_face_index = None

    # 检查面片索引是否越界
    face_index_valid = True
    if num_faces > 0:
        if min_face_index < 1 or max_face_index > num_vertices:
            face_index_valid = False
            invalid_face_indices += 1

    print("\n========== OBJ 检测结果 ==========")
    print(f"OBJ file: {obj_path}")
    print(f"Number of vertices: {num_vertices}")
    print(f"Number of faces: {num_faces}")
    print(f"Minimum face index: {min_face_index}")
    print(f"Maximum face index: {max_face_index}")
    print(f"Face index valid: {face_index_valid}")
    print(f"Non-triangle faces: {non_triangle_faces}")
    print(f"Invalid vertex lines: {invalid_vertex_lines}")
    print(f"Invalid face lines: {invalid_face_lines}")
    print(f"Has vertex color: {has_vertex_color}")
    print(f"Vertex color lines: {vertex_color_count}")

    print("\n========== 结论 ==========")
    if (
        num_vertices > 0
        and num_faces > 0
        and face_index_valid
        and non_triangle_faces == 0
        and invalid_vertex_lines == 0
        and invalid_face_lines == 0
    ):
        print("OBJ 文件格式正常：顶点数量和面片数量有效，面片索引合法，所有面片均为三角面片。")
    else:
        print("OBJ 文件存在格式问题，需要检查上面的错误信息。")


if __name__ == "__main__":
    check_obj("outputs/emma_all_faces/face_000/geometry/face.obj")
    check_obj("outputs/emma_all_faces/face_001/geometry/face.obj")
    check_obj("outputs/emma_all_faces/face_002/geometry/face.obj")
    check_obj("outputs/backsun2_res/face_000/geometry/face.obj")