from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.figure(figsize=(10, 10))

for digit in range(10):
    indices = np.where(y_train == digit)[0][:5]  # 해당 숫자 5개 추출
    for j, idx in enumerate(indices):
        plt_idx = digit * 5 + j + 1
        plt.subplot(10, 5, plt_idx)
        plt.imshow(x_train[idx], cmap='gray')
        plt.axis('off')
        if j == 0:
            plt.title(f"Label {digit}")

plt.tight_layout()
plt.show()
