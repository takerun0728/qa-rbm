from PIL import Image
import numpy as np
from scipy import linalg

COMP_RGB1 = 360
COMP_RGB2 = 240
COMP_RGB3 = 120
COMP_RGB4 = 48

COMP_YUV1 = 180
COMP_YUV2 = 120
COMP_YUV3 = 60
COMP_YUV4 = 24



#r = np.empty((480, 640))
#g = np.empty((480, 640))
#b = np.empty((480, 640))
#r[:,:] = np.linspace(0, 255, 640)
#g[:,::-1] = np.linspace(0, 255, 640)
#b[:,:] = np.linspace(0, 255, 480).reshape(-1, 1)
#org_img = Image.fromarray(np.array([r, g, b], dtype=np.uint8).transpose(1, 2, 0))
#org_img.save('org.bmp')

org_img = Image.open('pen.jpeg')
tmp_org = np.array(org_img)
r = tmp_org[:, :, 0]
g = tmp_org[:, :, 1]
b = tmp_org[:, :, 2]
u_r, s_r, v_r = np.linalg.svd(r, full_matrices=False)
u_g, s_g, v_g = np.linalg.svd(g, full_matrices=False)
u_b, s_b, v_b = np.linalg.svd(b, full_matrices=False)
r_comp1 = u_r[:, :COMP_RGB1] @ np.diag(s_r[:COMP_RGB1]) @ v_r[:COMP_RGB1, :]
g_comp1 = u_g[:, :COMP_RGB1] @ np.diag(s_g[:COMP_RGB1]) @ v_g[:COMP_RGB1, :]
b_comp1 = u_b[:, :COMP_RGB1] @ np.diag(s_b[:COMP_RGB1]) @ v_b[:COMP_RGB1, :]
r_comp2 = u_r[:, :COMP_RGB2] @ np.diag(s_r[:COMP_RGB2]) @ v_r[:COMP_RGB2, :]
g_comp2 = u_g[:, :COMP_RGB2] @ np.diag(s_g[:COMP_RGB2]) @ v_g[:COMP_RGB2, :]
b_comp2 = u_b[:, :COMP_RGB2] @ np.diag(s_b[:COMP_RGB2]) @ v_b[:COMP_RGB2, :]
r_comp3 = u_r[:, :COMP_RGB3] @ np.diag(s_r[:COMP_RGB3]) @ v_r[:COMP_RGB3, :]
g_comp3 = u_g[:, :COMP_RGB3] @ np.diag(s_g[:COMP_RGB3]) @ v_g[:COMP_RGB3, :]
b_comp3 = u_b[:, :COMP_RGB3] @ np.diag(s_b[:COMP_RGB3]) @ v_b[:COMP_RGB3, :]
r_comp4 = u_r[:, :COMP_RGB4] @ np.diag(s_r[:COMP_RGB4]) @ v_r[:COMP_RGB4, :]
g_comp4 = u_g[:, :COMP_RGB4] @ np.diag(s_g[:COMP_RGB4]) @ v_g[:COMP_RGB4, :]
b_comp4 = u_b[:, :COMP_RGB4] @ np.diag(s_b[:COMP_RGB4]) @ v_b[:COMP_RGB4, :]
comp_rgb1_img = Image.fromarray(np.array([r_comp1.clip(0, 255), g_comp1.clip(0, 255), b_comp1.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_rgb2_img = Image.fromarray(np.array([r_comp2.clip(0, 255), g_comp2.clip(0, 255), b_comp2.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_rgb3_img = Image.fromarray(np.array([r_comp3.clip(0, 255), g_comp3.clip(0, 255), b_comp3.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_rgb4_img = Image.fromarray(np.array([r_comp4.clip(0, 255), g_comp4.clip(0, 255), b_comp4.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_rgb1_img.save('comp_rgb1.bmp')
comp_rgb2_img.save('comp_rgb2.bmp')
comp_rgb3_img.save('comp_rgb3.bmp')
comp_rgb4_img.save('comp_rgb4.bmp')

y = 0.299 * r + 0.587 * g + 0.114 * b
u = -0.169 * r - 0.331 * g + 0.500 * b
v = 0.500 * r - 0.419 * g - 0.081 * b
u_y, s_y, v_y = np.linalg.svd(y, full_matrices=False)
u_u, s_u, v_u = np.linalg.svd(u, full_matrices=False)
u_v, s_v, v_v = np.linalg.svd(v, full_matrices=False)
y_comp1 = u_y[:, :] @ np.diag(s_y[:]) @ v_y[:, :]
u_comp1 = u_u[:, :COMP_YUV1] @ np.diag(s_u[:COMP_YUV1]) @ v_u[:COMP_YUV1, :]
v_comp1 = u_v[:, :COMP_YUV1] @ np.diag(s_v[:COMP_YUV1]) @ v_v[:COMP_YUV1, :]
y_comp2 = u_y[:, :COMP_YUV2 * 4] @ np.diag(s_y[:COMP_YUV2 * 4]) @ v_y[:COMP_YUV2 * 4, :]
u_comp2 = u_u[:, :COMP_YUV2] @ np.diag(s_u[:COMP_YUV2]) @ v_u[:COMP_YUV2, :]
v_comp2 = u_v[:, :COMP_YUV2] @ np.diag(s_v[:COMP_YUV2]) @ v_v[:COMP_YUV2, :]
y_comp3 = u_y[:, :COMP_YUV3 * 4] @ np.diag(s_y[:COMP_YUV3 * 4]) @ v_y[:COMP_YUV3 * 4, :]
u_comp3 = u_u[:, :COMP_YUV3] @ np.diag(s_u[:COMP_YUV3]) @ v_u[:COMP_YUV3, :]
v_comp3 = u_v[:, :COMP_YUV3] @ np.diag(s_v[:COMP_YUV3]) @ v_v[:COMP_YUV3, :]
y_comp4 = u_y[:, :COMP_YUV4 * 4] @ np.diag(s_y[:COMP_YUV4 * 4]) @ v_y[:COMP_YUV4 * 4, :]
u_comp4 = u_u[:, :COMP_YUV4] @ np.diag(s_u[:COMP_YUV4]) @ v_u[:COMP_YUV4, :]
v_comp4 = u_v[:, :COMP_YUV4] @ np.diag(s_v[:COMP_YUV4]) @ v_v[:COMP_YUV4, :]
r_comp1 = y_comp1 + 1.402 * v_comp1
g_comp1 = y_comp1 - 0.344 * u_comp1 - 0.714 * v_comp1
b_comp1 = y_comp1 + 1.772 * u_comp1
r_comp2 = y_comp2 + 1.402 * v_comp2
g_comp2 = y_comp2 - 0.344 * u_comp2 - 0.714 * v_comp2
b_comp2 = y_comp2 + 1.772 * u_comp2
r_comp3 = y_comp3 + 1.402 * v_comp3
g_comp3 = y_comp3 - 0.344 * u_comp3 - 0.714 * v_comp3
b_comp3 = y_comp3 + 1.772 * u_comp3
r_comp4 = y_comp4 + 1.402 * v_comp4
g_comp4 = y_comp4 - 0.344 * u_comp4 - 0.714 * v_comp4
b_comp4 = y_comp4 + 1.772 * u_comp4

comp_yuv1_img = Image.fromarray(np.array([r_comp1.clip(0, 255), g_comp1.clip(0, 255), b_comp1.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_yuv2_img = Image.fromarray(np.array([r_comp2.clip(0, 255), g_comp2.clip(0, 255), b_comp2.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_yuv3_img = Image.fromarray(np.array([r_comp3.clip(0, 255), g_comp3.clip(0, 255), b_comp3.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_yuv4_img = Image.fromarray(np.array([r_comp4.clip(0, 255), g_comp4.clip(0, 255), b_comp4.clip(0, 255)], dtype=np.uint8).transpose(1, 2, 0))
comp_yuv1_img.save('comp_yuv1.bmp')
comp_yuv2_img.save('comp_yuv2.bmp')
comp_yuv3_img.save('comp_yuv3.bmp')
comp_yuv4_img.save('comp_yuv4.bmp')

pass