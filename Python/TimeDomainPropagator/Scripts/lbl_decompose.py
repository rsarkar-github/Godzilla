import numpy as np


def lbl_decompose_f2(A):
    """
    :param A: symmetric boolean matrix with zero diagonal
    :return: L,B for A = L B L^T
    """

    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise TypeError("Matrix not square")

    if np.all(np.logical_xor(A.transpose(), A)):
        raise ValueError("Matrix not symmetric")

    L = np.identity(n=n, dtype=np.int32)
    B = np.dot(L, A)

    for i in range(n-1):

        i1 = i
        for j in range(n):
            if B[i, j] == 1:
                i1 = j
                break
        if i1 == i:
            continue

        for j in range(i+1, n):
            if B[j, i1] == True:
                E = np.identity(n=n, dtype=np.int32)
                E[j, i] = 1
                B = np.dot(E, B) % 2
                B = np.dot(B, E.transpose()) % 2
                L = np.dot(L, E) % 2

        # print("i=", i)
        # print("B=\n", B)

    return L, B


def create_pauli_list(A):

    n = A.shape[0]
    L, B = lbl_decompose_f2(A)

    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if B[i,j] == 1:
                pairs.append((i, j))
                continue

    n_qubits = len(pairs)
    P = np.zeros(shape=(n, 2 * n_qubits), dtype=np.int32)

    for index, item in enumerate(pairs):
        P[item[0], index] = 1
        P[item[1], index + n_qubits] = 1

    return np.dot(L, P) % 2


if __name__ == "__main__":

    # # Anticommuting list
    # n = 7
    # A = np.zeros(shape=(n, n), dtype=np.int32)
    # A += 1
    # for i in range(n):
    #     A[i, i] = 0
    # print("A = \n", A)
    #
    # P = create_pauli_list(A)
    # print("\nNumber of qubits = ", int(P.shape[1] / 2))
    # print("\nPauli list = \n", P)
    #
    # # CAL list
    # n = 10
    # A = np.zeros(shape=(n, n), dtype=np.int32)
    # for i in range(n):
    #     A[i, (i + 1) % n] = 1
    #     A[i, (i - 1) % n] = 1
    # print("A = \n", A)
    #
    # P = create_pauli_list(A)
    # print("\nNumber of qubits = ", int(P.shape[1] / 2))
    # print("\nPauli list = \n", P)

    # Generate random matrix
    n = 15
    A = np.zeros(shape=(n, n), dtype=np.int32)

    n1 = 100
    for i in range(n1):
        j1 = np.random.randint(0, n)
        j2 = np.random.randint(0, n)
        A[j1, j2] += 1
        A[j2, j1] += 1

    # Add ones to diagonal randomly
    for i in range(n1):
        j1 = np.random.randint(0, n)
        A[j1, j1] += 1

    # # Zero out the diagonal
    # for i in range(n):
    #     A[i, i] = 0

    A = A % 2
    print("A = \n", A)

    L, B = lbl_decompose_f2(A)
    print("B = \n", B)
