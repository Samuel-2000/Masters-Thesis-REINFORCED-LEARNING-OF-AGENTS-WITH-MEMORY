T_vert = np.array([
    [-1,  0, -1],
    [ 1,  0,  1],
    [-1,  0, -1]
    ], dtype=np.int8)
T_horiz = np.array([
    [-1,  1, -1],
    [ 0,  0,  0],
    [-1,  1, -1]
    ], dtype=np.int8)

T_diag1 = np.array([
    [-1,  0,  1],
    [ 0,  0,  0],
    [ 1,  0, -1]
    ], dtype=np.int8)
T_diag2 = np.array([
    [ 1,  0, -1],
    [ 0,  0,  0],
    [-1,  0,  1]
    ], dtype=np.int8)


T_a = np.array([
    [-1,  1, -1],
    [ 0,  0, -1],
    [ 1,  0, -1]
    ], dtype=np.int8)
T_b = np.array([
    [-1, -1, -1],
    [ 0,  0,  1],
    [ 1,  0, -1]
    ], dtype=np.int8)

T_c = np.array([
    [-1,  0,  1],
    [ 1,  0,  0],
    [-1, -1, -1]
    ], dtype=np.int8)
T_d = np.array([
    [-1,  0,  1],
    [-1,  0,  0],
    [-1,  1, -1]
    ], dtype=np.int8)

T_e = np.array([
    [-1,  1, -1],
    [-1,  0,  0],
    [-1,  0,  1]
    ], dtype=np.int8)
T_f = np.array([
    [-1, -1, -1],
    [ 1,  0,  0],
    [-1,  0,  1]
    ], dtype=np.int8)


T_g = np.array([
    [ 1,  0, -1],
    [ 0,  0, -1],
    [-1,  1, -1]
    ], dtype=np.int8)
T_h = np.array([
    [ 1,  0, -1],
    [ 0,  0,  1],
    [-1, -1, -1]
    ], dtype=np.int8)