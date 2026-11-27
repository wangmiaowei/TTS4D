"""Static benchmarks of the supplementary: Mip-NeRF 360, Tanks and Temples (3DGS hold-out) and DL3DV-140.

Mip-NeRF 360 / Tanks and Temples (``<root>/<scene>/{images*, sparse/0/{cameras,images}.bin}``; Tanks and Temples = the
``tandt`` folder released with 3D Gaussian Splatting):
  3DGS evaluation split (scene/dataset_readers.py, llffhold 8): images sorted by name, every 8th image (index % 8 == 0) is a
  target, the others form the input split; ``N`` inputs = ``uniform`` over the input split (``N = 0``: all of them).
  Resolution of the 3DGS evaluation (full_eval.py): Mip-NeRF 360 outdoor scenes ``images_4``, indoor scenes ``images_2``,
  Tanks and Temples ``images`` (979 x 546).  Cameras from the COLMAP model (OpenCV w2c -> c2w), PINHOLE intrinsics rescaled
  to the image size.  Metric protocol ``3dgs``.
  Scene unit (``unit='input'``, the paper's static evaluation): seq_scale = 1 / max(span_in, floor), span_in = largest
  distance of an input camera from the first input, floor = 20th percentile of span_in over the scenes of the dataset (same
  N).  ``unit='sequence'``: span over all images of the scene instead (used by the active view selection experiment).

DL3DV-140 (Long-LRM / MVP benchmark): the 140 scenes of ``dl3dv_benchmark.txt``, targets = every 8th frame (0, 8, ...),
  inputs = the K-means input indices of ``dl3dv_fold_8_kmeans_input_idx.json`` (both files from the MVP repository), images =
  ``images_4`` (960 x 540) of the DL3DV-10K release (zip or folder), undistorted with the OpenCV model of the benchmark's
  ``nerfstudio/transforms.json`` (cv2.getOptimalNewCameraMatrix alpha 0), cameras from the same file (nerfstudio -> OpenCV).
  Scene unit: seq_scale = 1 / max(span of the whole sequence, 8.7679).  Metric protocol ``default``.
"""
import glob
import json
import os
import struct
import zipfile

import numpy as np

from .common import assemble, read_rgb, resize_short, span, uniform

MIP_OUTDOOR = ["bicycle", "flowers", "garden", "stump", "treehill"]
MIP_INDOOR = ["room", "counter", "kitchen", "bonsai"]
TANDT = ["truck", "train"]
DL3DV_SPAN_FLOOR = 8.7679
LLFFHOLD = 8


# ------------------------------------------------------------------------------------------------ COLMAP
def _read_cameras(path):
    npar = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
    names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE"}
    cams = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            p = np.array(struct.unpack("<" + "d" * npar[mid], f.read(8 * npar[mid])))
            cams[cid] = dict(model=names.get(mid, str(mid)), w=int(w), h=int(h), params=p)
    return cams


def _read_images(path):
    ims = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid = struct.unpack("<i", f.read(4))[0]
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            ims[iid] = dict(q=q, t=t, cam=cid, name=name.decode())
    return ims


def _qvec2rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def colmap_scene(root, scene):
    """-> names (sorted), c2w float32 [n, 4, 4] (OpenCV), K float32 [n, 4] in pixels of ``images/``, (h, w) of ``images/``."""
    import cv2
    r = f"{root}/{scene}"
    cams = _read_cameras(f"{r}/sparse/0/cameras.bin")
    ims = {v["name"]: v for v in _read_images(f"{r}/sparse/0/images.bin").values()}
    names = sorted(os.listdir(f"{r}/images"))
    assert sorted(ims) == names, f"{scene}: COLMAP images != image files"
    h, w = cv2.imread(f"{r}/images/{names[0]}").shape[:2]
    c2w, K = np.zeros((len(names), 4, 4)), np.zeros((len(names), 4))
    for i, n in enumerate(names):
        v, cam = ims[n], cams[ims[n]["cam"]]
        assert cam["model"] in ("PINHOLE", "SIMPLE_PINHOLE"), cam["model"]
        p = cam["params"]
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if cam["model"] == "SIMPLE_PINHOLE" else p[:4]
        sx, sy = w / cam["w"], h / cam["h"]
        K[i] = [fx * sx, fy * sy, cx * sx, cy * sy]
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = _qvec2rot(v["q"]), v["t"]
        c2w[i] = np.linalg.inv(w2c)
    return names, c2w.astype(np.float32), K.astype(np.float32), (h, w)


def split_3dgs(n):
    """(input split, targets) of the 3DGS evaluation (llffhold 8) for n name-sorted images."""
    return [i for i in range(n) if i % LLFFHOLD != 0], [i for i in range(n) if i % LLFFHOLD == 0]


def image_folder(dataset, scene):
    return "images" if dataset == "tandt" else ("images_4" if scene in MIP_OUTDOOR else "images_2")


def scenes_3dgs(root, dataset):
    want = TANDT if dataset == "tandt" else MIP_OUTDOOR + MIP_INDOOR
    return [s for s in want if os.path.isdir(f"{root}/{s}/sparse/0")]


def span_floor(root, dataset, n_in, unit="input"):
    """20th percentile of the scene spans over the scenes of the dataset found under ``root``."""
    sp = []
    for s in scenes_3dgs(root, dataset):
        names, c2w, _, _ = colmap_scene(root, s)
        if unit == "input":
            I = uniform(split_3dgs(len(names))[0], n_in or 10 ** 9)
            sp.append(span(c2w[I]))
        else:
            sp.append(round(span(c2w), 4))
    return float(np.percentile(sp, 20))


def build_3dgs(root, dataset, scene, n_in, floor, unit="input", short=None, inputs=None, targets=None):
    """One scene of the 3DGS hold-out.  ``n_in = 0``: every image of the input split.  ``inputs`` / ``targets`` (image
    indices) override the split (active view selection).  ``short``: resize to this short side (INTER_AREA)."""
    names, c2w, K, (h0, w0) = colmap_scene(root, scene)
    folder = image_folder(dataset, scene)
    train, test = split_3dgs(len(names))
    I = list(inputs) if inputs is not None else uniform(train, n_in or 10 ** 9)
    Q = list(targets) if targets is not None else test
    sp = span(c2w[I]) if unit == "input" else round(span(c2w), 4)     # whole-sequence spans were stored to 1e-4
    seq_scale = 1.0 / max(sp, floor)

    def view(i, group=None):
        im = read_rgb(f"{root}/{scene}/{folder}/{names[i]}")
        h, w = im.shape[:2]
        if w > 1600:
            raise ValueError("3DGS would downscale this image to width 1600: not supported")
        v = {"img": im, "c2w_world": c2w[i], "K": K[i] * np.array([w / w0, h / h0, w / w0, h / h0]), "t": 0.0, "frame": i, "cam": i}
        if group:
            v["group"] = group
        return v
    views = resize_short([view(i) for i in I] + [view(i, "test") for i in Q], short)
    meta = dict(dataset=dataset, scene=scene, protocol="3dgs_holdout", metric="3dgs", n_in_requested=int(n_in), unit=unit, span_floor=float(floor),
                rule="targets = name-sorted images [::8]; inputs uniform over the others", image_folder=folder, short_side=short,
                image_names_in=[names[i] for i in I], image_names_tgt=[names[i] for i in Q])
    return assemble(views, len(I), seq_scale, True, meta)


# ------------------------------------------------------------------------------------------------ DL3DV-140
def dl3dv_scenes(benchmark_list):
    """Scene hashes in the order of MVP's data/dl3dv_benchmark.txt."""
    return [ln.strip().split("/")[-2] for ln in open(benchmark_list) if ln.strip()]


def _dl3dv_images(images_root, h):
    """Reader of images_4/<name> for scene h: a DL3DV-10K zip (<images_root>/**/<h>.zip) or a folder <images_root>/<h>/."""
    if os.path.isdir(f"{images_root}/{h}"):
        return lambda name: open(f"{images_root}/{h}/images_4/{name}", "rb").read()
    zs = glob.glob(f"{images_root}/**/{h}.zip", recursive=True)
    if not zs:
        raise FileNotFoundError(f"no images for DL3DV scene {h} under {images_root}")
    z = zipfile.ZipFile(zs[0])
    prefix = next(n.split("images_4/")[0] for n in z.namelist() if "images_4/frame_" in n)
    return lambda name: z.read(f"{prefix}images_4/{name}")


def build_dl3dv(benchmark_root, images_root, kmeans_json, h, n_in):
    import cv2
    m = json.load(open(f"{benchmark_root}/{h}/nerfstudio/transforms.json"))
    frames = m["frames"]
    T = len(frames)
    km = {k["scene_name"]: k for k in json.load(open(kmeans_json))}
    I = sorted(int(i) for i in km[h][f"fold_8_kmeans_{n_in}_input"])
    Q = list(range(0, T, 8))
    assert not set(I) & set(Q)
    dist = np.asarray([m["k1"], m["k2"], m["p1"], m["p2"]], np.float64)
    read = _dl3dv_images(images_root, h)

    def c2w_of(fr):
        c = np.asarray(fr["transform_matrix"], np.float64).copy()
        c[0:3, 1:3] *= -1
        c = c[[1, 0, 2, 3], :]
        c[2, :] *= -1
        return c
    c2w_all = np.stack([c2w_of(fr) for fr in frames])
    seq_scale = 1.0 / max(span(c2w_all), DL3DV_SPAN_FLOOR)

    def view(i, group=None):
        img = cv2.imdecode(np.frombuffer(read(os.path.basename(frames[i]["file_path"])), np.uint8), cv2.IMREAD_COLOR)
        hh, ww = img.shape[:2]
        intr = np.asarray([[m["fl_x"] / m["w"] * ww, 0, m["cx"] / m["w"] * ww], [0, m["fl_y"] / m["h"] * hh, m["cy"] / m["h"] * hh], [0, 0, 1]])
        new, _ = cv2.getOptimalNewCameraMatrix(intr, dist, (ww, hh), 0, (ww, hh))
        img = cv2.cvtColor(cv2.undistort(img, intr, dist, None, new), cv2.COLOR_BGR2RGB)
        K = np.array([new[0, 0], new[1, 1], new[0, 2], new[1, 2]], np.float64)
        if (hh, ww) != (540, 960):                     # a few scenes come from 1080p sources (images_4 = 480 x 270): bilinear to 960 x 540
            img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_LINEAR)
            K = K * np.array([960 / ww, 540 / hh, 960 / ww, 540 / hh])
        v = {"img": img, "c2w_world": c2w_all[i], "K": K, "t": 0.0, "frame": i, "cam": i}
        if group:
            v["group"] = group
        return v
    views = [view(i) for i in I] + [view(i, "test") for i in Q]
    meta = dict(dataset="dl3dv", scene=h, protocol="dl3dv140", metric="default", n_in_requested=int(n_in),
                rule="inputs = MVP K-means indices; targets = every 8th frame")
    return assemble(views, len(I), seq_scale, True, meta)
