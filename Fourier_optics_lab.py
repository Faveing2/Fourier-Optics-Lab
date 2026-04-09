from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

image = Image.open("Airy_pattern.png")
experiment_image = Image.open("Slide 33 No Filter.png")
lpf_experiment_image = Image.open("LowPass Slide 33 Dot F.png")
hpf_experiment_image = Image.open("High  Pass Slide 33 Dot G 2.png")

data = np.array(image)
experiment_data = np.array(experiment_image)
lpf_experiment_data = np.array(lpf_experiment_image)
hpf_experiment_data = np.array(hpf_experiment_image)

data = data[..., :3].mean(axis=2)
fft_data = fft2(data)
fft_data = fftshift(fft_data)
fig, axs = plt.subplots(nrows=4, ncols=2)

experiment_data = experiment_data[..., :3].mean(axis=2)
ex_fft_data = fft2(experiment_data)
ex_fft_data = fftshift(ex_fft_data)

lpf_experiment_data = lpf_experiment_data[..., :3].mean(axis=2)
ex_lpf_fft_data = fft2(lpf_experiment_data)
ex_lpf_fft_data = fftshift(ex_lpf_fft_data)

hpf_experiment_data = hpf_experiment_data[..., :3].mean(axis=2)
ex_hpf_fft_data = fft2(hpf_experiment_data)
ex_hpf_fft_data = fftshift(ex_hpf_fft_data)

ny, nx = data.shape
y, x = np.indices((ny,nx))

cy, cx = ny//2, nx//2
r = np.sqrt((x - cx)**2 + (y - cy)**2)

#r_cutoff  = 9 # Airy_patter
r_cutoff = 9
mask = r <= r_cutoff
# sigma = 6  # controls blur strength
# mask = np.exp(-(r**2) / (2 * sigma**2))
F_filtered = fft_data * mask

filtered_image = ifft2(F_filtered)
filtered_image = np.abs(filtered_image)

plt.title("Airy Pattern Low Pass Filter")

# norm = np.linalg.norm(fft_data)
# unit_vector = fft_data / norm

#### Calculated
axs[0,0].imshow(data)
axs[1,0].imshow(np.log1p(np.abs(fft_data)))
axs[2,0].imshow(np.log1p(np.abs(F_filtered)))
axs[3,0].imshow(filtered_image)

axs[0,1].imshow(experiment_data)
axs[1,1].imshow(np.log1p(np.abs(ex_fft_data)))
axs[2,1].imshow(np.log1p(np.abs(ex_lpf_fft_data)))
axs[3,1].imshow(lpf_experiment_data)
#axs[1].title("Low Pass Filter")
plt.show()
plt.clf()

### Now lets do the HPF
r_cutoff = 9
hpf_mask = r >= r_cutoff
# sigma = 6  # controls blur strength
# mask = np.exp(-(r**2) / (2 * sigma**2))
F_filtered_hpf = fft_data * hpf_mask

filtered_image_hpf = ifft2(F_filtered_hpf)
filtered_image_hpf = np.abs(filtered_image_hpf)

fig, axs = plt.subplots(nrows=4, ncols=2)
axs[0,0].imshow(data)
axs[1,0].imshow(np.log1p(np.abs(fft_data)))
axs[2,0].imshow(np.log1p(np.abs(F_filtered_hpf)))
axs[3,0].imshow(filtered_image_hpf)

axs[0,1].imshow(experiment_data)
axs[1,1].imshow(np.log1p(np.abs(ex_fft_data)))
axs[2,1].imshow(np.log1p(np.abs(ex_hpf_fft_data)))
axs[3,1].imshow(hpf_experiment_data)

plt.show()

plt.clf()

### Lets try radially profiling
fig, axs = plt.subplots(nrows=2, ncols=2)

def get_radial_profile(data):
    ny, nx = data.shape
    y, x = np.indices((ny, nx))
    cy, cx = ny//2, nx//2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())

    radial_profile = tbin / nr

    return radial_profile

# Assume image is 2D
axs[0,0].plot(get_radial_profile(data))
axs[1,0].plot(get_radial_profile(filtered_image_hpf))
axs[0,1].plot(get_radial_profile(experiment_data))
axs[1,1].plot(get_radial_profile(hpf_experiment_data))
axs[0,0].title.set_text("Airy Pattern")
axs[1,0].title.set_text("Air Pattern HPF")
axs[0,1].title.set_text("Source Signal")
axs[1,1].title.set_text("Source Signal HPF")
plt.show()
plt.clf()

fig, axs = plt.subplots(nrows=2, ncols=2)

axs[0,0].plot(get_radial_profile(data))
axs[1,0].plot(get_radial_profile(filtered_image))
axs[0,1].plot(get_radial_profile(experiment_data))
axs[1,1].plot(get_radial_profile(lpf_experiment_data))
axs[0,0].title.set_text("Airy Pattern")
axs[1,0].title.set_text("Air Pattern LPF")
axs[0,1].title.set_text("Source Signal")
axs[1,1].title.set_text("Source Signal LPF")
plt.show()
plt.clf()

### Lets plot the radial patterns on the same figures

plt.plot(get_radial_profile(experiment_data), label="Source")
plt.plot(get_radial_profile(lpf_experiment_data), label="Filtered")
plt.title("LPF")
plt.show()
plt.clf()

plt.plot(get_radial_profile(experiment_data), label="Source")
plt.plot(get_radial_profile(hpf_experiment_data), label="Filtered")
plt.title("hPF")
plt.show()

plt.clf()

### Compare calculation to Simulation
r1 = np.linspace(0,1, len(get_radial_profile(data)))
r2 = np.linspace(0,1, len(get_radial_profile(experiment_data)))

plt.plot(r1, get_radial_profile(data), label="Air Pattern")
plt.plot(r2, get_radial_profile(experiment_data), label="Reconstruction")
plt.title("Airy Patter vs Reconstructed")

plt.show()
plt.clf()

r1 = np.linspace(0,1, len(get_radial_profile(filtered_image_hpf)))
r2 = np.linspace(0,1, len(get_radial_profile(hpf_experiment_data)))

plt.plot(r1, get_radial_profile(filtered_image_hpf), label="Air Pattern")
plt.plot(r2, get_radial_profile(hpf_experiment_data), label="Reconstruction")

plt.show()
plt.clf()

### lets find peaks

### Compare calculation to Simulation
r1 = np.linspace(0,1, len(get_radial_profile(data)))
r2 = np.linspace(0,1, len(get_radial_profile(experiment_data)))

profile_smooth = gaussian_filter1d(experiment_data, sigma=2)
peaks, _ = find_peaks(get_radial_profile(profile_smooth), prominence=0.1)

point_x = []
point_y = []
for peak in peaks:
    x_point = r2[peak]
    point_x.append(x_point)
    point_y.append(get_radial_profile(experiment_data)[peak])

plt.plot(r1, get_radial_profile(data), label="Air Pattern")
plt.plot(r2, get_radial_profile(experiment_data), label="Reconstruction")
plt.scatter(x=point_x,y=point_y)
plt.title("Airy Pattern vs Reconstructed")

plt.show()
plt.clf()

print(peaks)