import argparse
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from scipy.ndimage import maximum_filter

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

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


def main():
    parser = argparse.ArgumentParser()

    def _check_path(x: str) -> Path:
        p = Path(x)
        if not p.is_file():
            raise argparse.ArgumentTypeError("Input is not a file")
        return p

    parser.add_argument("--image_path", required=True, type=_check_path)
    parser.add_argument("--yaml_path", required=True, type=_check_path)
    parser.add_argument("--use-channel", type=int, default=0)
    args, _ = parser.parse_known_args()

    image_path: Path = args.image_path
    yaml_path: Path = args.yaml_path

    filter_factory = {
        "LoG": create_log_filter,
        "TilG": create_ellipsoid_filter,
        "DoG": create_dog_filter,
    }
    if not yaml_path.suffix.endswith(".yaml"):
        parser.error("input yaml is not a vaild yaml file")
    yaml = YAML()
    parameters = yaml.load(yaml_path.open("r"))
    print(yaml_path)
    print(parameters)
    try:
        filter_kws = parameters["Filter"]
        localmaxi_kws = parameters["LocalMaxima"]
    except KeyError as e:
        parser.error(f"input yaml is not a vaild yaml file: {e}")

    print(filter_kws)
    print(localmaxi_kws)
    _, filter_fn = filter_factory[filter_kws.pop("filter_type")](**filter_kws)
    output_results = []
    ch = f"c{args.use_channel:d}"
    with h5py.File(image_path, "r") as handler:
        keys = sorted(handler.keys(), key=lambda x: int(x[1:]))
        total = len(keys)

        # Sanity test for HDF dataset
        if not keys:
            parser.error(
                "No data found: HDF must contains Group named as follow t0 ... tn "
            )
        grp = handler[keys[0]]
        assert (
            isinstance(grp, h5py.Group) and ch in grp.keys()
        ), f"Invalid group or channel number: {args.use_channel}"

        for k in keys:
            # Retrieve  time from key value
            t = int(k[1:])
            # Process one frame at a time
            dset = handler[f"{k}/{ch}"]
            assert isinstance(dset, h5py.Dataset), "Sanity test"
            image = np.ascontiguousarray(dset).astype("f4")
            filtered_im = filter_fn(image)

            peaks, peak_values = detect_local_maxima_3d(
                filtered_im,
                **localmaxi_kws,
            )
            # Time estimation.
            n_peaks = peak_values.size
            if n_peaks > 0:
                t_indices = np.full((n_peaks, 1), fill_value=t, dtype=peaks.dtype)
                peaks = np.hstack((t_indices, peaks))
            else:
                # We expanded the empty peaks to size of (0, 4)
                peaks = np.empty((0, 4), dtype=peaks.dtype)

            print(
                f"Found {peak_values.size:d} peaks at t={t:d} ({t+1:d}/{total:d}) [{image.shape}]"
            )
            res = np.hstack([peaks, peak_values[:, None]])
            output_results.append(res)

    output_arr = np.concatenate(output_results)
    object_id = np.arange(output_arr.shape[0])[:, None]
    output_arr_with_id = np.hstack([object_id, output_arr])
    header = "object_id,t,z,y,x,peak_values"
    np.savetxt(
        image_path.with_name(image_path.stem + "_peaks.csv"),
        output_arr_with_id,
        delimiter=",",
        header=header,
        comments="",
        fmt="%d",
        newline="\n",
    )


if __name__ == "__main__":
    main()
