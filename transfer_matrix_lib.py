def netp(x, l):
    return 2 * popcnt(x) - l

def rol(x, n, l):
    return (x << n | x >> (l - n)) & (0xFFFFFFFFFFFFFFFF >> (64 - l))

def popcnt(x):
    return bin(x).count("1")

def vx(a, l, j, k, h):
    return j * netp(a ^ rol(a, 1, l), l) - k * j * netp(a ^ rol(a, 2, l), l) - h * netp(a, l)

def vz(a1, a2, l, j):
    return j * netp(a1 ^ a2, l)