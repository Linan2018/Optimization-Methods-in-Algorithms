import numpy as np


def inv(a):
    """逆"""
    a_ = np.linalg.inv(a)
    return a_


def eig(a):
    """特征值"""
    e_vals, e_vecs = np.linalg.eig(a)
    return e_vals, e_vecs


def svd(a):
    u, s, vh = np.linalg.svd(a)
    return u, s, vh


if __name__ == '__main__':
    a = np.array([[1,2],[3,4]])
    a_ = np.linalg.inv(a)
    e_vals, e_vecs = np.linalg.eig(a)
    u, s, vh = np.linalg.svd(a)
    print("逆", a_, sep='\n')
    print("特征值", e_vals, sep='\n')
    print("特征向量", e_vecs, sep='\n')
    print("奇异值", s, sep='\n')
    print("左奇异", u, sep='\n')
    print("右奇异", vh, sep='\n')


