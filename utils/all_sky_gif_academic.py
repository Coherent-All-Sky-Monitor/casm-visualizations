import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.colors as mcolors
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import astropy.units as u
from pygdsm import GlobalSkyModel
import builtins
import h5py
from matplotlib.patches import Ellipse
import matplotlib as mpl

plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.borderpad'] = 0.3
plt.rcParams['legend.labelspacing'] = 0.3
plt.rcParams['legend.handletextpad'] = 0.3
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'DejaVu Serif'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['font.size'] = 18
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams["axes.linewidth"] = 1.5

# --- 1. Setup and Configuration ---

NSIDE = 256
G = GlobalSkyModel()

location = EarthLocation(lon=-71.1097 * u.deg, lat=42.3736 * u.deg, height=10 * u.m)
now_utc = Time.now()
N_FRAMES = 144*3
TIME_SPAN_HOURS = 24
time_deltas = np.linspace(0, TIME_SPAN_HOURS, N_FRAMES) * u.hour
observation_times = now_utc + time_deltas

def find_index(redshift, redshifts):
    index = np.where(np.round(redshifts,3)==np.round(redshift,3))[0][0]
    return index

colors_blue = ['black', 'dodgerblue', 'lightskyblue', 'white']
n_bins = 1000
cmap_blue = mcolors.LinearSegmentedColormap.from_list('blue', colors_blue, N=n_bins)

def get_ralf_map(plot=False):
    name_continuous = '/Users/liamconnor/Downloads/Konietzka2025_DMmap_continuous_v1.hdf5'
    with h5py.File(name_continuous, 'r') as file:
        DM_continuous = file['DMvalues'][:]
        z_continuous  = file['redshifts'][:]

    name_fullsky = '/Users/liamconnor/Downloads/Konietzka2025_DMmap_fullsky1_v1.hdf5'
    with h5py.File(name_fullsky, 'r') as file:
        DM_fullsky = file['DMvalues'][:]
        z_fullsky  = file['redshifts'][:]

    redshift_plot = 0.01
    idx_plot = find_index(redshift_plot, z_fullsky)
    DM_plot = DM_fullsky[idx_plot, :]

    if plot:
        hp.mollview(DM_plot, unit=r'DM [pc cm$^{-3}$]', min=int(0), 
                    max=int(100), xsize=800, cmap=cmap_blue, title='')
        plt.gca().set_title('')
        ax = plt.gca()
        ellipse = Ellipse(xy=(0, 0), width=4, height=2, edgecolor='black', 
                          facecolor='none', linewidth=1.5, transform=ax.transData)
        ax.add_patch(ellipse)
        plt.show()

    return DM_plot

sky_map = get_ralf_map(plot=False)
nside_ralf = hp.npix2nside(len(sky_map))
sky_map = hp.ud_grade(sky_map, nside_out=NSIDE)

yellow_star_frames_remaining = 0
yellow_star_position = None
YELLOW_STAR_DURATION = 3.5
RED_STAR_MEAN = 4

OVERLAY_ALPHA = 0.4

# --- 2. Figure & GridSpec layout ---

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[2.2, 1.0],
    height_ratios=[1, 1],  # bottom row is only 25% as tall as top row
    wspace=0.15,
    hspace=0.10
)

# Right column axes (created once)
ax_bar  = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[1, 1])  # <-- NOTE: typo fixed below; keep reading


# Define the field of view disc parameters
CENTER_LON = 0
CENTER_LAT = 0
RADIUS_DEG = 80
OVERLAY_COLOR = 'gray'
OVERLAY_ALPHA = 0.6

center_vec = hp.ang2vec(CENTER_LON, CENTER_LAT, lonlat=True)
radius_rad = np.pi/2  # 90 deg

npix = hp.nside2npix(NSIDE)
pixels_in_disc = hp.query_disc(NSIDE, center_vec, radius_rad, nest=False)
alpha_mask = np.zeros_like(sky_map) + 0.25
alpha_mask[pixels_in_disc] = 0.50
pixels_in_disc = hp.query_disc(NSIDE, center_vec, 0.9*radius_rad, nest=False)
alpha_mask[pixels_in_disc] = 0.75

# --- 3. Precompute bar/hist series & create artists ONCE ---

rate_casm_bar = 1
rate_trad_bar = 0.1
rng_bar = np.random.default_rng(42)
casm_bar_counts = [0]
trad_bar_counts = [0]
casm_bar_increments = rng_bar.poisson(rate_casm_bar, size=N_FRAMES)
trad_bar_increments = rng_bar.poisson(rate_trad_bar, size=N_FRAMES)
for i in range(N_FRAMES):
    casm_bar_counts.append(casm_bar_counts[-1] + casm_bar_increments[i])
    trad_bar_counts.append(trad_bar_counts[-1] + trad_bar_increments[i])

# Create bar artists
categories = ["CASM", "Traditional\nFRB telescope"]
bars = ax_bar.bar(categories, [0, 0], color=["red", "goldenrod"])
ax_bar.set_title("Local Universe FRBs", fontsize=15)
ax_bar.set_ylabel("Number", fontsize=15, labelpad=10)
ax_bar.set_ylim(0, builtins.max(casm_bar_counts[-1], trad_bar_counts[-1]) * 1.10)
ax_bar.set_xticklabels(categories, fontsize=15)  # adjust fontsize as needed
bar_labels = [ax_bar.text(b.get_x()+b.get_width()/2., 0, "0", ha="center", va="bottom") for b in bars]

# Histogram series
lambda_total_casm = 0.5
lambda_total_trad = 5.0
r_min, r_max = 1.0, 10000.0
n_bins = 120
scale_casm, scale_trad = 50.0, 500.0
rng_hist = np.random.default_rng(12345)

edges = np.linspace(r_min, r_max, n_bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

shape_casm = centers**2 * np.exp(-centers / scale_casm)
shape_trad = centers**2 * np.exp(-centers / scale_trad)
shape_casm = shape_casm / shape_casm.sum() * lambda_total_casm
shape_trad = shape_trad / shape_trad.sum() * lambda_total_trad

inc_casm = rng_hist.poisson(shape_casm, size=(N_FRAMES, n_bins))
inc_trad = rng_hist.poisson(shape_trad, size=(N_FRAMES, n_bins))

hist_casm = np.zeros(n_bins, dtype=int)
hist_trad = np.zeros(n_bins, dtype=int)

(line_casm,) = ax_hist.plot([], [], drawstyle="steps-mid", label="CASM", color="red", lw=2)
(line_trad,) = ax_hist.plot([], [], drawstyle="steps-mid", label="Traditional FRB\ntelescope", color="goldenrod", lw=2)
ax_hist.set_xscale("log")
ax_hist.set_xlim(r_min*5, r_max)
ax_hist.set_ylim(0, 1)  # will expand
ax_hist.set_xlabel("Distance (Mpc)", fontsize=15)
ax_hist.set_ylabel("Number", fontsize=15, labelpad=10)
ax_hist.legend(loc="upper left", fontsize=8)

# --- 4. Build the left Healpy axes ONCE and lock it to the left GridSpec cell ---

def map_at_time(t):
    """Rotate the sky_map into local AltAz for time t and return the map."""
    theta_altaz, phi_altaz = hp.pix2ang(NSIDE, np.arange(npix), nest=False)
    altaz_frame = AltAz(obstime=t, location=location)
    local_coords = SkyCoord(az=(phi_altaz * u.rad),
                            alt=((np.pi/2.0 - theta_altaz) * u.rad),
                            frame=altaz_frame)
    gal = local_coords.transform_to('galactic')
    theta_galactic = np.pi/2.0 - gal.b.rad
    phi_galactic = gal.l.rad
    source_pix = hp.ang2pix(NSIDE, theta_galactic, phi_galactic, nest=False)
    return sky_map[source_pix]

# Initial map for frame 0
rotated_map0 = map_at_time(observation_times[0])

# Create a *temporary* axes in the left cell to steal its rectangle, then delete it
ax_left_dummy = fig.add_subplot(gs[:, 0])
left_rect = ax_left_dummy.get_position()
ax_left_dummy.remove()

L = left_rect  # Mollweide's rectangle in figure coords
total_h = 0.75 * L.width             # desired combined height for right panels
y_center = L.y0 + L.height / 2.0    # center vertically on the Mollweide
y0 = y_center - total_h / 2.0       # bottom of the stacked right panels

# Use the current right-column x, width from GridSpec as a starting point
rbox = ax_bar.get_position()
x0, w = rbox.x0, rbox.width

# Split the total height between bar (top) and hist (bottom)
vpad = 0.05                         # small gap between the two (figure coords)
h1 = (total_h - vpad) * 0.55        # top panel height (55%)
h2 = (total_h - vpad) * 0.45        # bottom panel height (45%)

# Position them (figure coordinates: [left, bottom, width, height])
ax_hist.set_position([x0, y0,                 w, h2])
ax_bar.set_position( [x0, y0 + h2 + vpad,     w, h1])

# Create healpy axes and move it to the left cell rectangle
hp.mollview(rotated_map0, nest=False, min=0, max=int(100), cbar=False,
            alpha=alpha_mask, hold=False, flip='astro', xsize=800,
            cmap=cmap_blue, title='', notext=True, fig=fig.number)
ax_left = plt.gca()                   # this is the healpy Mollweide axes
ax_left.set_position(left_rect)       # pin to GridSpec cell
plt.sca(ax_left)

# --- 5. Animation update ---

BRIGHT=False

def update(frame_index):
    global BRIGHT, yellow_star_frames_remaining, yellow_star_position, hist_casm, hist_trad

    current_time = observation_times[frame_index]
    rotated_map = map_at_time(current_time)

    # LEFT: redraw healpy without stacking
    # plt.sca(ax_left)
    # ax_left.clear()
    # hp.mollview(rotated_map, nest=False, min=0, max=int(100), cbar=False,
    #             alpha=alpha_mask, hold=False, flip='astro', xsize=800,
    #             cmap=cmap_blue, title='', notext=True, fig=fig.number,
    #             reuse_axes=True)

    fig.sca(ax_left)          # <- use the Figure method, not plt.sca
    ax_left.clear()
    ax_left.axis('off')
    hp.mollview(
        rotated_map,
        nest=False,
        min=0, max=int(100),
        cbar=False,
        alpha=alpha_mask,
        hold=False,
        flip='astro',
        xsize=800,
        cmap=cmap_blue,
        title='',
        notext=True,
        fig=fig,              # <- pass the Figure, not fig.number
        reuse_axes=True
    )

    # Traditional small circle (example)
    traditional_center_theta = np.pi/2.0
    traditional_center_phi = -1.
    traditional_radius = 0.075
    n_trad_points = 60
    angles = np.linspace(0, 2*np.pi, n_trad_points)
    trad_circle_theta = traditional_center_theta + traditional_radius*np.cos(angles)
    trad_circle_phi   = traditional_center_phi   + traditional_radius*np.sin(angles)
    hp.projplot(trad_circle_theta, trad_circle_phi, 'C1-', linewidth=2, alpha=0.8)

    # Yellow star persistence
    if yellow_star_frames_remaining > 0:
        hp.projplot(yellow_star_position[0], yellow_star_position[1], 'r*', markersize=17.5)
        yellow_star_frames_remaining -= 1
    elif np.random.randint(0, 15) == 8:
        bright_pixels_mask = rotated_map[pixels_in_disc] > 50
        bright_pixels_in_disc = pixels_in_disc[bright_pixels_mask]
        if len(bright_pixels_in_disc) > 0:
            selected_pixel = bright_pixels_in_disc[np.random.randint(0, len(bright_pixels_in_disc))]
            theta, phi = hp.pix2ang(NSIDE, selected_pixel, nest=False)
            yellow_star_position = (theta, phi)
            yellow_star_frames_remaining = int(YELLOW_STAR_DURATION) - 1
            hp.projplot(theta, phi, 'r*', markersize=17.5)

    # Red stars
    N_red = np.random.poisson(RED_STAR_MEAN)
    bright_pixels_mask = rotated_map[pixels_in_disc] > 0
    bright_pixels_in_disc = pixels_in_disc[bright_pixels_mask]
    if len(bright_pixels_in_disc) > 0 and N_red > 0:
        for _ in range(N_red):
            selected_pixel = bright_pixels_in_disc[np.random.randint(0, len(bright_pixels_in_disc))]
            theta, phi = hp.pix2ang(NSIDE, selected_pixel, nest=False)
            hp.projplot(theta, phi, 'y*', markersize=5, alpha=0.90)

    # Field-of-view circle
    n_circle_points = 360
    circle_theta, circle_phi = [], []
    for angle in np.linspace(0, 2*np.pi, n_circle_points):
        if abs(center_vec[2]) < 0.9:
            v1 = np.cross(center_vec, [0, 0, 1])
        else:
            v1 = np.cross(center_vec, [1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(center_vec, v1)
        v2 = v2 / np.linalg.norm(v2)
        circle_point = (center_vec * np.cos(radius_rad) +
                        np.sin(radius_rad) * (v1 * np.cos(angle) + v2 * np.sin(angle)))
        theta_point = np.arccos(circle_point[2])
        phi_point = np.arctan2(circle_point[1], circle_point[0])
        if phi_point < 0:
            phi_point += 2*np.pi
        circle_theta.append(theta_point)
        circle_phi.append(phi_point)
    hp.projplot(circle_theta, circle_phi, 'k-', linewidth=1.25, alpha=0.8)

    # Text labels
    hp.projtext(np.pi/2.0, +2.35, "below horizon",
                fontsize=14, color='k', weight='bold', ha='center', va='center')
    hp.projtext(np.pi/1.75, -1.05, "Traditional\n telescope",
                fontsize=8, color='C1', weight='bold', ha='center', va='center')
    hp.projtext(np.pi/7.0, 0.0, "CASM-256",
                fontsize=13.5, color='white', weight='bold', ha='center', va='center')

    # RIGHT TOP: update bars
    if frame_index < len(casm_bar_counts):
        current_casm_bar = casm_bar_counts[frame_index]
        current_trad_bar = trad_bar_counts[frame_index]
        bars[0].set_height(current_casm_bar)
        bars[1].set_height(current_trad_bar)
        bar_labels[0].set_y(current_casm_bar); bar_labels[0].set_text(f"{int(current_casm_bar)}")
        bar_labels[1].set_y(current_trad_bar); bar_labels[1].set_text(f"{int(current_trad_bar)}")

    # RIGHT BOTTOM: update histogram
    # if frame_index < N_FRAMES:
    #     hist_casm += inc_casm[frame_index]
    #     hist_trad += inc_trad[frame_index]
    #     line_casm.set_data(centers, hist_casm)
    #     line_trad.set_data(centers, hist_trad)
    #     current_max = builtins.max(int(hist_casm.max()), int(hist_trad.max()))
    #     ymin, ymax = ax_hist.get_ylim()
    #     if current_max > 0.95 * ymax:
    #         ax_hist.set_ylim(0, current_max * 1.10)

    # RIGHT BOTTOM: update histogram
    if frame_index < N_FRAMES:
        hist_casm += inc_casm[frame_index]
        hist_trad += inc_trad[frame_index]

        # pad both ends so the steps touch y=0 at start & end
        x_pad = np.r_[centers[0], centers, centers[-1]]
        y_casm = np.r_[0,        hist_casm, 0]
        y_trad = np.r_[0,        hist_trad, 0]

        line_casm.set_data(x_pad, y_casm)   # drawstyle="steps-mid" already set
        line_trad.set_data(x_pad, y_trad)

        current_max = builtins.max(int(hist_casm.max()), int(hist_trad.max()))
        ymin, ymax = ax_hist.get_ylim()
        if current_max > 0.95 * ymax:
            ax_hist.set_ylim(0, current_max * 1.10)

    return ()

# --- 6. Animate & Save ---

print("Generating animation... This may take a few minutes.")
ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False)

try:
    ani.save('combined_animation.gif', writer=PillowWriter(fps=12), dpi=100)
    print("\nAnimation saved successfully as 'combined_animation.gif'")
except Exception as e:
    print(f"\nCould not save animation. Error: {e}")
    print("Showing interactive plot instead.")
    plt.show()
