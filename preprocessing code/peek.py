import numpy as np, matplotlib.pyplot as plt

ct = np.load("your-ct-npz-path-here")[np.load("your-ct-npz-path-here").files[0]]

print(f"Shape: {ct.shape} | dtype: {ct.dtype} | min: {ct.min():.2f} | max: {ct.max():.2f} | mean: {ct.mean():.2f}")

plt.imshow(ct[ct.shape[0]//2], cmap="gray")
plt.title("Middle slice preview")
plt.axis("off")
plt.show()
