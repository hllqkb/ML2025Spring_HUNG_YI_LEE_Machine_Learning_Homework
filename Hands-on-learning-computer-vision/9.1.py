import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# 1. 加载 SAM 模型
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 2. 读取图片
image_path = "cat.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# 3. matplotlib 交互点击
click_points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        click_points.append((x, y))
        print(f"点击位置: {(x, y)}")
        ax.plot(x, y, 'ro')  # 红点
        fig.canvas.draw()

fig, ax = plt.subplots()
ax.imshow(image_rgb)
ax.set_title("点击选择目标，关闭窗口结束")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# 4. 生成 SAM 掩码
if len(click_points) == 0:
    raise ValueError("未选择任何点！")

input_point = np.array(click_points)
input_label = np.ones(len(click_points))  # 1 表示前景点

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)

best_mask = masks[np.argmax(scores)]

# 5. 提取前景
mask_uint8 = (best_mask.astype(np.uint8) * 255)
mask_3ch = cv2.merge([mask_uint8, mask_uint8, mask_uint8])
segmented_image = cv2.bitwise_and(image, mask_3ch)

# 6. 显示结果
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(best_mask, cmap='gray')
plt.title("Selected Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title("Segmented Foreground")
plt.axis('off')

plt.tight_layout()
plt.show()
