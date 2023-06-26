import os

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from rascal.calibrator import Calibrator
from rascal.atlas import Atlas
from rascal import util

# Load data
pixelsDe, spectrum = np.loadtxt("d.csv", delimiter=',', dtype='U').T

spectrum = spectrum.astype('float32')

# Identify the peaks
peaks, _ = find_peaks(spectrum, prominence=10, distance=5, height=95)
peaks = util.refine_peaks(spectrum, peaks, window_width=5)

# Initialise the calibrator
c = Calibrator(peaks, spectrum=spectrum)

c.set_hough_properties(
    num_slopes=2000,
    range_tolerance=100.0,
    xbins=20,
    ybins=20,
    min_wavelength=4000.0,
    max_wavelength=8000.0,
    linearity_tolerance=500,
)
c.set_ransac_properties(sample_size=10, top_n_candidate=7, ransac_tolerance=10,
                        minimum_matches=2,  minimum_fit_error=1e-3)


atlas_lines_nm = [ 404.6565, 407.7837, 35.8335, 491.6068, 546.075, 557.581, 576.961, 579.067, 587.096, 696.5431, 706.7218, 730.2936, 738.3980, 750.3869, 751.4652, 758.741,
                  760.155, 763.5106, 768.525, 769.454, 772.4207, 785.482, 794.8176, ] 

print("number of Lamp lines")
print(len(atlas_lines_nm))

number = 10
 
atlas_lines = []
 
for val in atlas_lines_nm:
   atlas_lines.append(val * number)

element = ["MY"] * len(atlas_lines)

atlas = Atlas(range_tolerance=9.0)
atlas.add_user_atlas(elements=element, wavelengths=atlas_lines,)

c.set_atlas(atlas,candidate_tolerance=200)

c.do_hough_transform()

# Run the wavelength calibration
(
    best_p,
    matched_peaks,
    matched_atlas,
    rms,
    residual,
    peak_utilisation,
    atlas_utilisation,
) = c.fit(max_tries=5000, fit_deg=3, candidate_tolerance=8)


# Plot the solution
c.plot_fit(
    best_p, spectrum, plot_atlas=True, log_spectrum=False, tolerance=2
)

# Show the parameter space for searching possible solution
c.plot_search_space()

print("Stdev error: {} A".format(residual.std()))
print("Peaks utilisation rate: {}%".format(peak_utilisation * 100))
print("Atlas utilisation rate: {}%".format(atlas_utilisation * 100))
