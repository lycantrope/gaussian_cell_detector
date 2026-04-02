import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import napari
import numpy as np
import torch
import torch.nn.functional as F
from magicgui import magicgui, use_app
from magicgui.widgets import Container, FileEdit, Label, LineEdit, PushButton, TextEdit
from napari.qt import thread_worker
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage import color, morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from tifffile import tifffile

DEVICE = torch.accelerator.current_accelerator() or torch.device("cpu")


def create_conv3d_filter(kernel: np.ndarray):
    w_z, w_y, w_x = tuple(i // 2 for i in kernel.shape)
    kernel_t = torch.from_numpy(kernel[None, None, ...].astype("f4")).to(DEVICE)

    def apply_filter(img, kernel=kernel_t):
        with torch.no_grad():
            img = torch.from_numpy(img)
            img = img.to(kernel.dtype)
            img = img.to(DEVICE)[None, None, ...]
            # Different from numpy, the padding in pytorch is inside out, so the padding order must be reversed.
            img_padded = F.pad(img, (w_x, w_x, w_y, w_y, w_z, w_z), mode="replicate")
            filtered = F.conv3d(img_padded, kernel, padding=0)
            return filtered[0, 0].cpu().numpy()

    return apply_filter


def create_ellipsoid_filter(
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
    w_x: int,
    w_y: int,
    w_z: int,
):
    # Apply TilG directly
    x = np.arange(-w_x, w_x + 1)
    y = np.arange(-w_y, w_y + 1)
    z = np.arange(-w_z, w_z + 1)

    # broadcasting to: [2*w_z+1, 2*w_y+1, 2*w_x+1]
    exponent = (
        (z**2 / (2 * sigma_z**2))[:, None, None]
        + (y**2 / (2 * sigma_y**2))[None, :, None]
        + (x**2 / (2 * sigma_x**2))[None, None, :]
    )

    f = np.exp(-exponent)
    N = f.size
    F1 = f.sum()
    F2 = (f * f).sum()
    denom = F2 * N - F1 * F1

    kernel = (N * f - F1) / denom

    return kernel, create_conv3d_filter(kernel)


def create_dog_filter(
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
    sigma2_x: float,
    sigma2_y: float,
    sigma2_z: float,
    w_x: int,
    w_y: int,
    w_z: int,
):
    x = np.arange(-w_x, w_x + 1)
    y = np.arange(-w_y, w_y + 1)
    z = np.arange(-w_z, w_z + 1)

    exp1 = (
        (z**2 / (2 * sigma_z**2))[:, None, None]
        + (y**2 / (2 * sigma_y**2))[None, :, None]
        + (x**2 / (2 * sigma_x**2))[None, None, :]
    )

    exp2 = (
        (z**2 / (2 * sigma2_z**2))[:, None, None]
        + (y**2 / (2 * sigma2_y**2))[None, :, None]
        + (x**2 / (2 * sigma2_x**2))[None, None, :]
    )

    g1 = np.exp(-exp1)
    g2 = np.exp(-exp2)
    # 正規化（重要：比較の公平性）
    g1 /= g1.sum()
    g2 /= g2.sum()

    kernel = g1 - g2

    # DC 成分除去（LoG と同等）
    kernel -= kernel.mean()
    return kernel, create_conv3d_filter(kernel)


def create_log_filter(
    sigma_x: float,
    sigma_y: float,
    sigma_z: float,
    w_x: int,
    w_y: int,
    w_z: int,
):
    # Apply TilG directly

    x = np.arange(-w_x, w_x + 1)
    y = np.arange(-w_y, w_y + 1)
    z = np.arange(-w_z, w_z + 1)

    r2 = (
        (z**2 / (sigma_z**2))[:, None, None]
        + (y**2 / (sigma_y**2))[None, :, None]
        + (x**2 / (sigma_x**2))[None, None, :]
    )
    kernel = (r2 - 3) * np.exp(-0.5 * r2)

    kernel -= kernel.mean()
    tot = np.abs(kernel).sum()
    kernel /= tot
    kernel = -kernel

    return kernel, create_conv3d_filter(kernel)


def create_valid_region_mask(
    shape: Sequence[int],
    w_x: int,
    w_y: int,
    w_z: int,
) -> np.ndarray:
    # shape is (D, H, W)
    assert len(shape) == 3, "shape must be length of 3. "
    d, h, w = shape

    # Create 1D range arrays
    z = np.arange(d)
    y = np.arange(h)
    x = np.arange(w)
    z_valid = (z >= w_z) & (z < d - w_z)
    y_valid = (y >= w_y) & (y < h - w_y)
    x_valid = (x >= w_x) & (x < w - w_x)
    # Use broadcasting to reshape the array into (D, H, W):
    # z: (D, 1, 1), y: (1, H, 1), x: (1, 1, W)
    mask = z_valid[:, None, None] & y_valid[None, :, None] & x_valid[None, None, :]
    return mask


# =========================
# local maxima
# =========================


def detect_local_maxima_3d(
    image: np.ndarray,
    min_distance: int,
    threshold_percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    detect_local_maxima_3d
    image: 3D numpy array
    topk: top k peaks to return
    threshold_percentile: e.g. 99.5
    """

    size = 2 * min_distance + 1
    max_filt = maximum_filter(image, size=size, mode="constant")

    threshold_abs = np.percentile(image, threshold_percentile)

    is_peak = image == max_filt
    mask = is_peak & (image >= threshold_abs)

    peaks = np.column_stack(np.nonzero(mask))
    peak_values = image[peaks[:, 0], peaks[:, 1], peaks[:, 2]]
    return peaks, peak_values


@dataclass(frozen=True)
class PeakResult:
    t: int
    peaks: np.ndarray
    peak_values: np.ndarray
    filtered_im: Optional[np.ndarray] = None
    model_im: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.peak_values.size

    @property
    def size(self) -> int:
        return len(self)


def get_file_list(first_file: Path):
    home = first_file.parent
    basename = first_file.name
    # 例: t012.tif → prefix='t', start=12, ext='.tif'
    regex = re.compile(r"(.*?)(\d+)(\.tif+)$")
    m = regex.match(basename)
    if m is None:
        raise ValueError("File name must end with a number, e.g. t0.tif")

    prefix, start_str, ext = m.groups()

    pattern = f"{prefix}*{ext}"

    filedict = {}
    for f in home.glob(pattern):
        m = regex.match(f.name)
        if m is None:
            continue
        # Exclude the files containing different prefix.
        pat, idx, ext = m.groups()
        if pat != prefix:
            continue

        filedict[int(idx)] = f

    files = [filedict[k] for k in sorted(filedict.keys())]

    if len(files) == 0:
        raise ValueError("No TIFF files found")
    assert first_file in files, "Sanity test, the files should contains first_file"
    return files


def gaussian_1d(w, sigma):
    # Create the 1D coordinate array
    x = np.arange(-w, w + 1)
    # 1D Gaussian kernel
    g = np.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()


def generate_model_image(
    image_shape,
    peaks,
    peak_values,
    sigma_x,
    sigma_y,
    sigma_z,
    w_x,
    w_y,
    w_z,
    **kwargs,  # This is for catch unuse arguments in DoG filters
):
    """
    image_shape: (z,y,x)
    peaks: (N,4) t,z,y,x
    peak_values: (N,)
    """

    # モデル楕円体（中心0）
    gx = gaussian_1d(w_x, sigma_x)
    gy = gaussian_1d(w_y, sigma_y)
    gz = gaussian_1d(w_z, sigma_z)
    base = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]

    Z, Y, X = image_shape
    dz, dy, dx = base.shape
    model_pad = np.zeros((Z + dz - 1, Y + dy - 1, X + dx - 1), dtype=np.float32)
    for (t0, z0, y0, x0), amp in zip(peaks, peak_values):
        model_pad[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx] += amp * base

    return model_pad[w_z : w_z + Z, w_y : w_y + Y, w_x : w_x + X]


def generate_label_image(
    filtered_im,
    peaks,
):
    """
    filtered_im at time t: (z,y,x)
    peaks: (N,4) t,z,y,x
    """
    # モデル楕円体（中心0）

    inverted = filtered_im.max() - filtered_im

    # 3. Apply Watershed
    # The mask ensures we only segment where the intensity is meaningful
    # Adjust this threshold based on your specific background noise level
    mask = filtered_im > (filtered_im.mean() + 2 * filtered_im.std())
    # 3. Define markers: find local maxima of the distance map
    # A common technique is thresholding the distance map to find cell centers
    markers = np.zeros_like(mask, dtype=int)
    for i, (_, z0, y0, x0) in enumerate(peaks, start=1):
        markers[z0, y0, x0] = i

    markers = morphology.dilation(markers, morphology.ball(1))
    # 4. Apply watershed
    # The function works on 3D arrays seamlessly
    labels = watershed(
        inverted,
        markers,
        mask=mask,
    )  # Note the negative distance to treat centers as minima

    return labels


def find_peak_all(
    start_t,
    im_4d,
    parameters,
    min_distance,
    threshold_percentile,
    show_filter: bool = False,
    show_model: bool = False,
):
    _, filter_fn = parameters["filter_fn"](**parameters["filter_kws"])
    for t in range(im_4d.shape[0]):
        # Process one frame at a time
        tic = time.perf_counter()

        filtered_im = filter_fn(im_4d[t])

        peaks, peak_values = detect_local_maxima_3d(
            filtered_im,
            min_distance,
            threshold_percentile,
        )
        # Time estimation.
        n_peaks = peak_values.size
        if n_peaks > 0:
            t_indices = np.full((n_peaks, 1), fill_value=t + start_t, dtype=peaks.dtype)
            peaks = np.hstack((t_indices, peaks))
        else:
            # We expanded the empty peaks to size of (0, 4)
            peaks = np.empty((0, 4), dtype=peaks.dtype)

        model_im = None
        if show_model and n_peaks > 0:
            # model_im = generate_model_image(
            #     filtered_im.shape,
            #     peaks,
            #     peak_values,
            #     **parameters["filter_kws"],
            # )
            model_im = generate_label_image(filtered_im, peaks)

        if not show_filter:
            filtered_im = None
        elapsed_t = time.perf_counter() - tic
        res = PeakResult(
            t=t + start_t,
            peaks=peaks,
            peak_values=peak_values,
            filtered_im=filtered_im,
            model_im=model_im,
        )
        yield res, elapsed_t


# =========================
# main
# =========================
def main():
    ### global variable
    # image data, current we loaded all images eagerly.
    images = None
    ## Parameters for gaussian filter and find peaks
    parameters = {}
    ## find peaks results
    results: Dict[int, PeakResult] = {}

    # status board for output messages
    status_board = TextEdit(label="Log")
    status_board.read_only = True

    def show_message(msg: str, end="\n"):
        print(msg, end=end)
        status_board.value = (
            f"[{datetime.now().strftime('%H:%M:%S')}] {msg}{end}{status_board.value}"
        )

    def clear_peak_results():
        # Erase previous results
        nonlocal results
        results = {}
        viewer = napari.current_viewer()
        if viewer is None:
            return
        for layer in ["peaks", "filtered", "model"]:
            if layer in viewer.layers:
                viewer.layers.remove(viewer.layers[layer])

    @magicgui(
        load_mode={"choices": ["TIFF stack", "TIFF sequence"]},
        no_of_frames={"min": 0, "max": 10000, "step": 1},
        first_file={"widget_type": FileEdit, "mode": "r", "filter": "*.tif *.tiff"},
        persist=True,
        call_button="Load Image",
    )
    def load_image_widget(
        load_mode="TIFF stack",
        no_of_frames: int = 10000,
        first_file: Path = Path.home(),
    ):
        """
        @magicgui(
            image_file={"widget_type": FileEdit, "mode": "r", "filter": "*.tif *.tiff"},
            call_button="Load image"
        )
        def load_image_widget(image_file=default_image_path):
            nonlocal img, img_t, image_layer, points_layer, image_dim
        """
        nonlocal images

        if not first_file.is_file() and first_file.suffix != ".tif":
            show_message(f"Selected file is not a valid TIFF file: {first_file.name}")
            return

        if load_mode == "TIFF stack":
            image_mmap = tifffile.memmap(first_file, mode="r")

            if image_mmap.ndim == 4:
                T = image_mmap.shape[0]
                images = image_mmap[: min(no_of_frames, T)]
            elif image_mmap.ndim == 3:
                # (Z, Y, X) => (1, Z, Y, X)
                images = image_mmap[None, ...]
            elif image_mmap.ndim == 2:
                # (Y, X) => (1, 1, Y, X)
                images = image_mmap[None, None, ...]
            else:
                raise ValueError(f"Unsupported shape : {image_mmap.shape}")

            images = np.ascontiguousarray(images).astype("f4")

            del image_mmap
        elif "TIFF sequence":
            filelist = get_file_list(first_file)
            # We read the file starts from the first_file
            start = filelist.index(first_file)
            end = min(start + no_of_frames, len(filelist))
            filelist = filelist[start:end]
            # check file shape:
            im = tifffile.imread(filelist[0])
            assert im.ndim < 4, "Tiff sequence does not support 4D stack (T, Z, Y, X)"

            img_seq = tifffile.TiffSequence(get_file_list(first_file))
            images = img_seq.asarray().astype("f4")
            if images.ndim == 3:
                # Read from 2D sequence.
                # (T, Y, X) => (T, 1, Y, X)
                images[:, None, ...]
            images = np.ascontiguousarray(images)
        else:
            return

        viewer = napari.current_viewer()
        if images is None or viewer is None:
            return
        if "original" in viewer.layers:
            viewer.layers.remove(viewer.layers["original"])

        scales = np.ones(images.ndim)
        # Either TZYX or ZYX should works
        scales[-3] = z_scale_widget.z_scale.value

        low, high = np.percentile(images, (0, 99))
        viewer.add_image(
            images,
            name="original",
            colormap="gray",
            scale=scales,
            contrast_limits=(low, high),
        )
        show_message(f"Image Loaded {images.shape}")
        clear_peak_results()

    @magicgui(
        auto_call=True,
        filter_type={"choices": ["LoG", "DoG", "TilG"]},
        sigma_xy={"min": 0.5, "max": 6.0, "step": 0.1},
        sigma_z={"min": 0.5, "max": 4.0, "step": 0.1},
        sigma2_xy={"min": 0.5, "max": 6.0, "step": 0.1},
        sigma2_z={"min": 0.5, "max": 6.0, "step": 0.1},
        w_xy={"min": 1, "max": 12, "step": 1},
        w_z={"min": 1, "max": 6, "step": 1},
        persist=True,
    )
    def filter_widget(
        filter_type="LoG",
        sigma_xy=2.0,
        sigma_z=1.0,
        sigma2_xy=3.5,
        sigma2_z=1.5,
        w_xy=4,
        w_z=1,
    ):
        nonlocal parameters

        filter_widget.sigma2_xy.enabled = False
        filter_widget.sigma2_z.enabled = False
        if filter_type == "TilG":
            parameters["filter_type"] = "TilG"
            parameters["filter_fn"] = create_ellipsoid_filter
            parameters["filter_kws"] = dict(
                sigma_x=sigma_xy,
                sigma_y=sigma_xy,
                sigma_z=sigma_z,
                w_x=w_xy,
                w_y=w_xy,
                w_z=w_z,
            )

        elif filter_type == "DoG":
            filter_widget.sigma2_xy.enabled = True
            filter_widget.sigma2_z.enabled = True
            parameters["filter_type"] = "DoG"
            parameters["filter_fn"] = create_dog_filter
            parameters["filter_kws"] = dict(
                sigma_x=sigma_xy,
                sigma_y=sigma_xy,
                sigma_z=sigma_z,
                sigma2_x=sigma2_xy,
                sigma2_y=sigma2_xy,
                sigma2_z=sigma2_z,
                w_x=w_xy,
                w_y=w_xy,
                w_z=w_z,
            )
        else:
            parameters["filter_type"] = "LoG"
            parameters["filter_fn"] = create_log_filter
            parameters["filter_kws"] = dict(
                sigma_x=sigma_xy,
                sigma_y=sigma_xy,
                sigma_z=sigma_z,
                w_x=w_xy,
                w_y=w_xy,
                w_z=w_z,
            )

    @magicgui(
        call_button="Find peaks",
        min_distance={"min": 1, "max": 5, "step": 1},
        threshold_percentile={"min": 80, "max": 99.99, "step": 0.01},
        mode={"choices": ["This frame", "All frames"]},
        running={
            "widget_type": "CheckBox",
            "enabled": False,
            "label": "Idle",
        },
        persist=True,
    )
    def find_peak_widget(
        min_distance: int,
        threshold_percentile: float = 99.5,
        mode: str = "This frame",
        show_filtered: bool = False,
        show_model: bool = False,
        running=False,
    ):
        # This one become main widget in the right side
        if running:
            show_message("Find Peak is running. Please wait!")
            return

        viewer = napari.current_viewer()
        if viewer is None or images is None:
            return

        nonlocal parameters
        if not parameters:
            filter_widget()

        if mode == "All frames" and images.ndim > 3:
            start_t = 0
            img_to_analyze = images
        else:
            start_t = viewer.dims.current_step[0]
            img_to_analyze = images[start_t]
            # Expand t dimension
            img_to_analyze = img_to_analyze[None, ...]

        assert isinstance(img_to_analyze, np.ndarray), "Sanity test for type checking"

        tic = time.perf_counter()

        def finished(*args, tic=tic, **kws):
            find_peak_widget.running.value = False
            find_peak_widget.running.text = "Idle"
            elapse = time.perf_counter() - tic
            show_message(f"Find Peaks Finished: {elapse:.2f}")
            viewer = napari.current_viewer()
            if viewer is not None:
                names = ["original", "filtered", "model", "peaks"]
                src_idx = [viewer.layers.index(n) for n in names if n in viewer.layers]
                viewer.layers.move_multiple(src_idx, 0)

        # Disable some button to make process safe.
        find_peak_widget.running.value = True
        find_peak_widget.running.text = "Busy"

        clear_peak_results()
        worker_fn = thread_worker(
            find_peak_all,
            connect={"yielded": update_point_layer, "returned": finished},
            start_thread=True,
            progress={"total": img_to_analyze.shape[0]},
        )

        is_single = mode == "This frame"
        worker_fn(
            start_t,
            img_to_analyze,
            parameters,
            min_distance,
            threshold_percentile,
            show_filtered and is_single,
            show_model and is_single,
        )
        parameters["LocalMaxima"] = {
            "min_distance": min_distance,
            "threshold_percentile": threshold_percentile,
        }
        show_message(f"Start Find Peaks")
        show_message(f"{parameters['filter_type']}")
        show_message(f"{parameters['filter_kws']}")

    # Custom callback to disable show_filtered and show_model in All Frame
    def _mode_callback():
        is_single = find_peak_widget.mode.value == "This frame"
        find_peak_widget.show_filtered.enabled = is_single
        find_peak_widget.show_model.enabled = is_single

    find_peak_widget.mode.changed.connect(_mode_callback)

    def update_point_layer(res: Tuple[PeakResult, float]):
        peak_res, toc = res
        if peak_res.size == 0:
            show_message(f"No Peaks found in t={peak_res.t}")
            return
        show_message(
            f"Found {peak_res.peak_values.size:d} peaks at t={peak_res.t:d} ({toc:.2f}s)"
        )

        # Append timestamp to peaks

        # Assign the result to global states
        nonlocal results
        results[peak_res.t] = peak_res

        # Update napari peaks
        viewer = napari.current_viewer()
        if viewer is None:
            return

        if "peaks" in viewer.layers:
            prev_peaks = viewer.layers["peaks"].data
            all_peaks = np.concatenate([prev_peaks, peak_res.peaks])
            viewer.layers["peaks"].data = all_peaks
        else:
            # insert peaks
            viewer.add_points(
                peak_res.peaks,
                name="peaks",
                size=4,
                face_color="cyan",
                scale=viewer.layers["original"].scale,
            )

        if peak_res.filtered_im is not None:
            if "filtered" in viewer.layers:
                viewer.layers.remove(viewer.layers["filtered"])
            # ZYX -> TZYX
            filtered_im = peak_res.filtered_im[None, ...]
            scales = np.ones(filtered_im.ndim)
            # Either TZYX or ZYX should works
            scales[-3] = z_scale_widget.z_scale.value

            contrast_limits = np.percentile(filtered_im, (0.0, 99.0))

            viewer.add_image(
                filtered_im,
                name="filtered",
                colormap="gray",
                scale=scales,
                contrast_limits=contrast_limits,
                translate=[peak_res.t, 0, 0, 0],
            )

        if peak_res.model_im is not None:
            if "model" in viewer.layers:
                viewer.layers.remove(viewer.layers["model"])

            # ZYX -> TZYX
            model_im = peak_res.model_im[None, ...]
            scales = np.ones(model_im.ndim)
            # Either TZYX or ZYX should works
            scales[-3] = z_scale_widget.z_scale.value

            contrast_limits = np.percentile(model_im, (0.0, 99.0))
            viewer.add_image(
                color.label2rgb(model_im),
                name="model",
                # colormap="gray",
                scale=scales,
                # contrast_limits=contrast_limits,
                translate=[peak_res.t, 0, 0, 0],
            )

    @magicgui(
        z_scale={
            "widget_type": "FloatSlider",
            "label": "z scale",
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
        },
        persist=True,
        auto_call=True,
        call_button=False,
    )
    def z_scale_widget(z_scale=1.0):
        viewer = napari.current_viewer()
        if viewer is None:
            return
        layers = viewer.layers
        names = [
            "original",
            "peaks",
            "filtered",
            "model",
        ]

        for name in names:
            if name in layers:
                scale = np.array(layers[name].scale)
                scale[-3] = z_scale
                layers[name].scale = scale

    # ===========================
    # Custom Widgets to Save File
    # ===========================
    save_result_labels = [
        LineEdit(value="", label="outputdir:"),
        LineEdit(value="", label="csvfile:"),
        LineEdit(value="", label="paramsfile:"),
    ]
    save_result_btn = PushButton(label="Save")

    @save_result_btn.changed.connect
    def _save_callback():
        # A custom callback that trigger dialog and save the result directly.
        nonlocal results
        if not results:
            show_message("No peak data to save")
            return

        show_file_dialog = use_app().get_obj("show_file_dialog")
        init_path = os.fspath(load_image_widget.first_file.value.parent)
        save_path = show_file_dialog(
            "w",
            caption="Select file",
            start_path=init_path,
            filter="*.csv",
        )
        if not save_path:
            return

        save_path = Path(save_path)

        filter_type = parameters["filter_type"]
        params_path = save_path.parent / (
            save_path.stem + f"-params-{filter_type}.yaml"
        )

        save_result_labels[0].set_value(str(save_path.parent))
        save_result_labels[1].set_value(save_path.name)
        save_result_labels[2].set_value(params_path.name)

        keys = sorted(results.keys())
        peaks = np.concatenate([results[k].peaks for k in keys])
        peak_values = np.concatenate([results[k].peak_values for k in keys])
        object_id = np.arange(len(peaks))
        peaks2 = np.hstack((object_id[:, None], peaks, peak_values[:, None]))
        peaks2 = peaks2.astype("u4")
        header = "object_id,t,z,y,x,peak_values"
        np.savetxt(
            save_path,
            peaks2,
            delimiter=",",
            header=header,
            comments="",
            fmt="%d",
            newline="\n",
        )

        # Save filter and local maximum conditions

        params_for_save = CommentedMap()
        params_for_save["Filter"] = {
            "filter_type": filter_type,
            **parameters["filter_kws"],
        }
        params_for_save["LocalMaxima"] = parameters["LocalMaxima"]

        yaml = YAML()
        with params_path.open("w", encoding="utf-8") as fd:
            yaml.dump(params_for_save, fd)

        show_message(f"Save results and params to: {save_path.parent}")

    save_result_widget = Container(
        widgets=[
            *save_result_labels,
            save_result_btn,
        ],
        labels=True,
    )

    # Init Default Status
    filter_widget()
    find_peak_widget.running.value = False
    find_peak_widget.running.text = "Idle"

    # main loop
    viewer = napari.Viewer(ndisplay=2)
    container = Container(
        widgets=[
            Container(
                widgets=[
                    Label(value="Load Image"),
                    load_image_widget,
                    z_scale_widget,
                ],
                labels=False,
            ),
            Container(
                widgets=[Label(value="Gaussian Filters"), filter_widget],
                labels=False,
            ),
            Container(
                widgets=[Label(value="Find Peaks"), find_peak_widget],
                labels=False,
            ),
            save_result_widget,
        ],
        labels=False,
    )

    viewer.window.add_dock_widget(
        container,
        area="right",
        name="GaussianCellDetector",
        tabify=True,
    )
    viewer.window.add_dock_widget(
        Container(
            widgets=[Label(value="Status"), status_board],
            labels=False,
        ),
        area="right",
        name="Log",
        tabify=True,
    )

    napari.run()


if __name__ == "__main__":
    main()
