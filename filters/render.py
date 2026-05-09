import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no viewer window

import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from typing import Optional
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ellipse_params(
    cov: np.ndarray,
    n_sigma: float = 2.0,
) -> tuple[float, float, float]:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)
    angle_deg = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    width  = 2.0 * n_sigma * np.sqrt(vals[1])
    height = 2.0 * n_sigma * np.sqrt(vals[0])
    return width, height, angle_deg


def _ellipse_aabb(
    center: np.ndarray,
    cov: np.ndarray,
    n_sigma: float = 2.0,
) -> tuple[float, float, float, float]:
    vals, vecs = np.linalg.eigh(cov)
    vals  = np.maximum(vals, 0.0)
    a     = n_sigma * np.sqrt(vals[1])
    b     = n_sigma * np.sqrt(vals[0])
    theta = np.arctan2(vecs[1, 1], vecs[0, 1])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = np.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
    dy = np.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)
    cx, cy = center
    return cx - dx, cx + dx, cy - dy, cy + dy


def _smooth_limit(
    current: float,
    target: float,
    expand_alpha: float,
    contract_alpha: float,
    expanding: bool,
) -> float:
    alpha = expand_alpha if expanding else contract_alpha
    return current + alpha * (target - current)


def _shade_verts(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if len(x) < 2:
        cx = float(x[0]) if len(x) else 0.0
        cy = 0.0
        return np.array([[cx, cy], [cx, cy], [cx, cy], [cx, cy]])
    return np.concatenate([
        np.column_stack([x, lower]),
        np.column_stack([x[::-1], upper[::-1]]),
    ])


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def render_history(
    true_pos: dict[str, np.ndarray],
    filter_pos: dict[str, np.ndarray],
    filter_uncertainties: dict[str, np.ndarray],
    filter_residuals: dict[str, np.ndarray],
    filter_residual_std_history: dict[str, np.ndarray],
    *,
    output_path: str = "simulation.mp4",
    fps: int = 30,
    tail_length: int = 150,
    padding: float = 2.0,
    n_sigma: float = 3.0,
    expand_alpha: float = 0.6,
    contract_alpha: float = 0.04,
    residual_cols: int = 2,
    traj_fig_height: float = 6.0,
    residual_row_height: float = 2.8,
    fig_width: float = 12.0,
    dpi: int = 150,
    bitrate: int = 3000,
) -> None:

    entity_names:   list[str] = list(true_pos.keys())
    filtered_names: set[str]  = set(filter_pos.keys())

    n_frames = max(len(true_pos[n]) for n in entity_names) - 1
    offset = n_frames / (max(len(v) for v in filter_residuals.values()))

    res_slots: list[tuple[str, int, str]] = []
    for name in entity_names:
        if name not in filtered_names or name not in filter_residuals:
            continue
        S = filter_residuals[name].shape[1]
        for d in range(S):
            res_slots.append((name, d, f"{name}  ·  Sensor {d + 1}"))

    n_res_plots   = len(res_slots)
    n_res_rows    = max(1, int(np.ceil(n_res_plots / residual_cols))) if n_res_plots else 0
    has_residuals = n_res_plots > 0

    cmap   = plt.cm.get_cmap("tab10", max(len(entity_names), 1))
    colors: dict[str, tuple] = {
        name: cmap(i) for i, name in enumerate(entity_names)
    }

    residual_block_height = n_res_rows * residual_row_height if has_residuals else 0.0
    fig_height = traj_fig_height + residual_block_height

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="#111118")

    if has_residuals:
        outer_gs = gridspec.GridSpec(
            2, 1,
            figure=fig,
            height_ratios=[traj_fig_height, residual_block_height],
            hspace=0.38,
        )
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            n_res_rows, residual_cols,
            subplot_spec=outer_gs[1],
            hspace=0.65,
            wspace=0.30,
        )
        traj_spec = outer_gs[0]
    else:
        outer_gs  = gridspec.GridSpec(1, 1, figure=fig)
        traj_spec = outer_gs[0]
        inner_gs  = None

    ax_traj = fig.add_subplot(traj_spec)

    def _style(ax: plt.Axes, xlabel: str = "", ylabel: str = "") -> None:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#44445a")
        ax.tick_params(colors="#aaaacc", labelsize=7.5)
        ax.grid(True, color="#2e2e4e", linewidth=0.5, linestyle="--")
        if xlabel:
            ax.set_xlabel(xlabel, color="#aaaacc", fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color="#aaaacc", fontsize=8)

    _style(ax_traj, xlabel="x", ylabel="y")
    ax_traj.set_title("Entity Trajectories", color="#ddddff", fontsize=12, pad=8)

    res_axes: list[plt.Axes] = []
    if has_residuals and inner_gs is not None:
        for idx in range(n_res_plots):
            row, col = divmod(idx, residual_cols)
            ax_r = fig.add_subplot(inner_gs[row, col])
            _style(ax_r, xlabel="Filter step", ylabel="Residuals")
            _, _, label = res_slots[idx]
            ax_r.set_title(label, color="#ccccee", fontsize=9,
                           fontweight="bold", pad=4)
            res_axes.append(ax_r)

        for spare in range(n_res_plots, n_res_rows * residual_cols):
            row, col = divmod(spare, residual_cols)
            ax_spare = fig.add_subplot(inner_gs[row, col])
            ax_spare.set_visible(False)

    true_lines:     dict[str, plt.Line2D]       = {}
    filter_lines:   dict[str, plt.Line2D]       = {}
    true_markers:   dict[str, plt.Line2D]       = {}
    filter_markers: dict[str, plt.Line2D]       = {}
    ellipses:       dict[str, mpatches.Ellipse] = {}
    legend_handles: list                        = []

    for name in entity_names:
        c = colors[name]

        tl, = ax_traj.plot([], [], "-",  color=c, linewidth=1.5, alpha=0.85, zorder=2)
        tm, = ax_traj.plot([], [], "o",  color=c, markersize=6,  zorder=4)
        true_lines[name]   = tl
        true_markers[name] = tm
        legend_handles.append(
            plt.Line2D([0], [0], color=c, linestyle="-", linewidth=2,
                       label=f"{name}  (true)")
        )

        if name in filtered_names:
            fl, = ax_traj.plot([], [], "--", color=c, linewidth=1.5, alpha=0.75, zorder=3)
            fm, = ax_traj.plot([], [], "s",  color=c, markersize=5,  zorder=5)
            filter_lines[name]   = fl
            filter_markers[name] = fm

            ell = mpatches.Ellipse(
                xy=(0, 0), width=1, height=1, angle=0,
                edgecolor=c, facecolor=(*c[:3], 0.08),
                linestyle="--", linewidth=1.2, zorder=1,
            )
            ell.set_visible(False)
            ax_traj.add_patch(ell)
            ellipses[name] = ell

            legend_handles.append(
                plt.Line2D([0], [0], color=c, linestyle="--", linewidth=2,
                           label=f"{name}  (filter ± {n_sigma:.0f}σ)")
            )

    ax_traj.legend(
        handles=legend_handles,
        loc="upper left",
        facecolor="#222233",
        edgecolor="#44445a",
        labelcolor="#ddddff",
        fontsize=8,
        framealpha=0.8,
    )

    res_scatter:     list[plt.Line2D] = []
    res_upper_lines: list[plt.Line2D] = []
    res_lower_lines: list[plt.Line2D] = []
    res_shade_polys: list[MplPolygon] = []

    for i, (name, d, _) in enumerate(res_slots):
        ax_r = res_axes[i]
        c    = colors[name]

        poly = MplPolygon(
            np.zeros((4, 2)),
            closed=True,
            facecolor=(*c[:3], 0.22),
            edgecolor="none",
            zorder=1,
        )
        ax_r.add_patch(poly)
        res_shade_polys.append(poly)

        ul, = ax_r.plot([], [], "--", color=(*c[:3], 0.65), linewidth=1.0, zorder=2)
        ll, = ax_r.plot([], [], "--", color=(*c[:3], 0.65), linewidth=1.0, zorder=2)
        res_upper_lines.append(ul)
        res_lower_lines.append(ll)

        sc, = ax_r.plot([], [], linestyle="none", marker="o",
                        color=c, markersize=3.5, alpha=0.9, zorder=3)
        res_scatter.append(sc)

    if res_axes:
        last_ax = res_axes[-1]
        last_ax.legend(
            handles=[
                plt.Line2D([0], [0], linestyle="none", marker="o",
                           color="#5599dd", markersize=5, label="Residuals"),
                mpatches.Patch(facecolor=(0.33, 0.6, 0.98, 0.30),
                               edgecolor="none",
                               label=f"{n_sigma:.0f}σ"),
            ],
            loc="upper right",
            facecolor="#222233",
            edgecolor="#44445a",
            labelcolor="#ddddff",
            fontsize=7.5,
            framealpha=0.8,
        )

    view: dict[str, Optional[float]] = {
        "xmin": None, "xmax": None,
        "ymin": None, "ymax": None,
    }

    def _update(frame: int):
        t          = frame
        tail_start = max(0, t - tail_length) if tail_length > 0 else 0

        req_x: list[float] = []
        req_y: list[float] = []

        for name in entity_names:
            tp = true_pos[name]
            true_lines[name].set_data(tp[tail_start : t + 1, 0],
                                      tp[tail_start : t + 1, 1])
            true_markers[name].set_data([tp[t, 0]], [tp[t, 1]])
            req_x.append(tp[t, 0])
            req_y.append(tp[t, 1])

            if name in filtered_names:
                fp  = filter_pos[name]
                cov = filter_uncertainties[name]
                t_f = min(t, len(fp) - 1)

                filter_lines[name].set_data(fp[tail_start : t_f + 1, 0],
                                            fp[tail_start : t_f + 1, 1])
                filter_markers[name].set_data([fp[t_f, 0]], [fp[t_f, 1]])

                cov_t     = cov[t_f]
                w, h, ang = _ellipse_params(cov_t, n_sigma)
                ell       = ellipses[name]
                ell.set_center((fp[t_f, 0], fp[t_f, 1]))
                ell.set_width(w);  ell.set_height(h);  ell.angle = ang
                ell.set_visible(True)

                x0, x1, y0, y1 = _ellipse_aabb(fp[t_f], cov_t, n_sigma)
                req_x.extend([x0, x1])
                req_y.extend([y0, y1])

        # Smooth axis limits with equal world-unit scaling.
        # We derive the required data ranges, then expand whichever axis is
        # "short" so that x and y have identical pixels-per-unit — matching
        # what set_aspect("equal") would do, but without the warning and
        # without constraining the axes box to be square.
        if req_x:
            data_xmin = min(req_x) - padding
            data_xmax = max(req_x) + padding
            data_ymin = min(req_y) - padding
            data_ymax = max(req_y) + padding

            x_span = data_xmax - data_xmin
            y_span = data_ymax - data_ymin

            # For equal scaling: y_span_world / x_span_world == _axes_ar
            # Expand whichever axis would otherwise be cramped.
            x_span_eq = max(x_span, y_span / _axes_ar)
            y_span_eq = x_span_eq * _axes_ar

            cx = (data_xmin + data_xmax) / 2
            cy = (data_ymin + data_ymax) / 2

            tgt_xmin = cx - x_span_eq / 2
            tgt_xmax = cx + x_span_eq / 2
            tgt_ymin = cy - y_span_eq / 2
            tgt_ymax = cy + y_span_eq / 2

            if view["xmin"] is None:
                view.update(xmin=tgt_xmin, xmax=tgt_xmax,
                            ymin=tgt_ymin, ymax=tgt_ymax)
            else:
                view["xmin"] = _smooth_limit(view["xmin"], tgt_xmin,
                    expand_alpha, contract_alpha, tgt_xmin < view["xmin"])
                view["xmax"] = _smooth_limit(view["xmax"], tgt_xmax,
                    expand_alpha, contract_alpha, tgt_xmax > view["xmax"])
                view["ymin"] = _smooth_limit(view["ymin"], tgt_ymin,
                    expand_alpha, contract_alpha, tgt_ymin < view["ymin"])
                view["ymax"] = _smooth_limit(view["ymax"], tgt_ymax,
                    expand_alpha, contract_alpha, tgt_ymax > view["ymax"])

            ax_traj.set_xlim(view["xmin"], view["xmax"])
            ax_traj.set_ylim(view["ymin"], view["ymax"])

        for i, (name, d, _) in enumerate(res_slots):
            if t % offset != 0: continue
            idx = int(t / offset)
            ax_r = res_axes[i]
            t_f  = min(idx, len(filter_residuals[name]) - 1)
            rs   = max(0, t_f - tail_length) if tail_length > 0 else 0

            x_idx = np.arange(rs, t_f + 1, dtype=float)
            res   = filter_residuals[name][rs : t_f + 1, d]
            std   = filter_residual_std_history[name][rs : t_f + 1, d]
            upper =  n_sigma * std
            lower = -n_sigma * std

            res_scatter[i].set_data(x_idx, res)
            res_upper_lines[i].set_data(x_idx, upper)
            res_lower_lines[i].set_data(x_idx, lower)
            res_shade_polys[i].set_xy(_shade_verts(x_idx, lower, upper))

            if len(res) > 0:
                all_y  = np.concatenate([res, upper, lower])
                ymin_r = float(all_y.min())
                ymax_r = float(all_y.max())
                margin = max((ymax_r - ymin_r) * 0.12, 0.1)
                ax_r.set_xlim(rs, max(rs + 1, t_f + 1))
                ax_r.set_ylim(ymin_r - margin, ymax_r + margin)

        return (
            list(true_lines.values())
            + list(true_markers.values())
            + list(filter_lines.values())
            + list(filter_markers.values())
            + list(ellipses.values())
            + res_scatter
            + res_upper_lines
            + res_lower_lines
            + res_shade_polys
        )

    # Finalise layout so get_position() reflects the real axes box dimensions,
    # then compute the axes aspect ratio (height / width, in inches).  This is
    # the y-world-units-per-x-world-unit ratio required for equal scaling.
    fig.canvas.draw()
    _pos  = ax_traj.get_position()           # fractions of the figure
    _fw, _fh = fig.get_size_inches()
    _axes_ar = (_pos.height * _fh) / (_pos.width * _fw)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000 / fps,
        blit=True,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"title": "Kalman Filter Simulation"},
        bitrate=bitrate,
    )

    with tqdm(total=n_frames, desc=f"Rendering → {output_path}",
              unit="frame", dynamic_ncols=True) as pbar:
        anim.save(
            output_path,
            writer=writer,
            dpi=dpi,
            progress_callback=lambda i, n: pbar.update(1),
        )
    plt.close(fig)
    print("Done.")
    