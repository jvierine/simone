import numpy as n
import numpy as np
import os
import h5py

def create_pseudo_random_code(clen=1000, seed=0):
    """
    seed is a way of reproducing the random code without
    having to store all actual codes. the seed can then
    act as a sort of station_id.
    """
    np.random.seed(seed)
    phases = np.array(
        np.sign(np.random.randn(clen)),
        dtype=np.complex64,
    )
    return(phases)

def periodic_convolution_matrix(envelope, rmin=0, rmax=100):
    """
    we imply that the number of measurements is equal to the number of elements
    in code

    """
    L = len(envelope)
    ridx = np.arange(rmin, rmax)
    A = np.zeros([L, rmax-rmin], dtype=np.complex64)
    for i in np.arange(L):
        A[i, :] = envelope[(i-ridx) % L]
    result = {}
    result['A'] = A
    result['ridx'] = ridx
    return(result)

def create_estimation_matrix(code, rmin=0, rmax=1000):
    r_cache = periodic_convolution_matrix(envelope=code, rmin=rmin, rmax=rmax)
    A = r_cache['A']
    Ah = np.transpose(np.conjugate(A))
    # least-squares estimate
    # B=(A^H A)^{-1}A^H
    B_cache = np.dot(np.linalg.inv(np.dot(Ah, A)), Ah)
    r_cache["code"]=code
    r_cache['B'] = B_cache
    return(r_cache)


def create_estimation_matrix_cached(seed,clen=1000,rmin=0,rmax=900):
    fname="b-%d-%d-%d.h5"%(seed,rmin,rmax)
    res={}
    if os.path.exists(fname):
        h=h5py.File(fname,"r")
        res["A"]=h["A"][()]
        # B=(A^H A)^{-1}A^H
        res["B"]=h["B"][()]
        res["code"]=h["code"][()]
        h.close()
    else:
        code=create_pseudo_random_code(clen=clen, seed=seed)
        res=create_estimation_matrix(code, rmin=rmin, rmax=rmax)
        h=h5py.File(fname,"w")
        h["A"]=res["A"]
        h["B"]=res["B"]
        h["code"]=res["code"]
        h.close()
    return(res)

# 
# k = 2*pi*f/vp = 2*pi*f*n/c = 2*pi*n/lambda_0
# |k_Bragg| = 4*pi*n/lambda_0 = 4*pi*sqrt(1-(f_p/f_r)^2)/lambda_0
# 

kb_code_seeds = n.array([5, 20, 608, 1686, 2755, 4972],dtype=int)
jr_code_seeds = n.array([7, 14, 137, 220, 419, 1796],dtype=int)


if __name__ == "__main__":
    res=create_estimation_matrix_cached(5,clen=1000,rmin=0,rmax=800)
    print(res)
