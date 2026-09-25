import numpy as np


def project_pag_on_mag(B_mag, B_bi_mag, B_pag, B_bi_pag):
    # B_pag follows the following convention:
    # B and B_bi
    # B[i,j] = 1 :::: i --> j :::: direct cause
    # B[i,j] = 2 ::::  i o-> j :::: direct cause or common confounder
    # B_bi[i,j] = 1 :::: i <-> j :::: common confounder
    # B_bi[i,j] = 2 :::: i --- j :::: selection bias - common effect
    # B_bi[i,j] = 3 :::: i o-o j :::: direct cause either direction or common confounder or common effect

    # B_mag follows the following convention:
    # B_mag[i,j] = 1 :::: i --> j :::: direct cause
    # B_bi_mag[i,j] = 1 ::::  i <-> j :::: common confounder
    # B_bi_mag[i,j] = 2 ::::  i --- j :::: selection bias - common effect

    n = B_mag.shape[0]

    R = np.zeros_like(B_mag)
    R_bi = np.zeros_like(B_bi_mag)
    for i in range(n):
        for j in range(n):
            if B_pag[i,j] == 2:
                if B_mag[i,j] == 1:
                    R[i, j] = 1
                elif B_bi_mag[i,j] == 1 or B_bi_mag[j,i] == 1:
                    R_bi[i,j] = 1
                    #R_bi[j,i] = 1
                else:
                    R[i,j] = 1 # Chosing arbitrarily direct causality
            elif R[i,j] == 0:
                R[i,j] = B_pag[i,j]

            if B_bi_pag[i,j] == 3:
                if B_bi_mag[i,j] == 1 or B_bi_mag[j,i] == 1 or B_bi_mag[i,j] == 2 or B_bi_mag[j,i] == 2:
                    R_bi[i,j] = B_bi_mag[i,j]
                    #R_bi[j,i] = B_bi_mag[j,i]
                elif B_mag[i,j] == 1:
                    R[i, j] = 1
                elif B_mag[j,i] == 1:
                    R[j, i] = 1
                else:
                    R_bi[i,j] = 1 # chosing arbitrarily confounder.
            elif R_bi[i,j] == 0:
                R_bi[i,j] = B_bi_pag[i,j]


    return R, R_bi
