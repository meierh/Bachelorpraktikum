import numpy as np


def compact1by2(x: np.uint32):
    x &= 0x09249249                    # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3    # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f    # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff    # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff   # x = ---- ---- ---- ---- ---- --98 7654 3210
    return x


# 32bit inverse Morton Code for 3D spaces
def inverse_3d_morton_code(rank: int):
    rank = np.uint32(rank)
    x = compact1by2(rank >> 0)
    y = compact1by2(rank >> 1)
    z = compact1by2(rank >> 2)

    return x, y, z
