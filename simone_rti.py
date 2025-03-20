import simone_config as sc
import digital_rf as drf
import numpy as n
import matplotlib.pyplot as plt
import stuffr
import scipy.signal.windows
import h5py


from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


#sc.kb_code_seeds# = n.array([5, 20, 608, 1686, 2755, 4972],dtype=int)
#sc.jr_code_seeds# = n.array([7, 14, 137, 220, 419, 1796],dtype=int)

kb=[]
jr=[]
nrg=500
for kbs in sc.kb_code_seeds:
    print(kbs)
    res=sc.create_estimation_matrix_cached(kbs,clen=1000,rmin=0,rmax=nrg)
    kb.append(res)
for jrs in sc.jr_code_seeds:
    res=sc.create_estimation_matrix_cached(jrs,clen=1000,rmin=0,rmax=nrg)
    jr.append(res)
d=drf.DigitalRFReader("/mnt/data/juha/spacex/20250219_reentry_event_Moitin/Moitin/20250219/")
b=d.get_bounds("ch000")

print(stuffr.unix2datestr(b[0]/100e3))
print(stuffr.unix2datestr(b[1]/100e3))
#exit(0)
L=100000
clen=1000
nt=int(L/clen)

RTI=n.zeros([2,nt,nrg],dtype=n.float32)
Z=n.zeros([2,nt,1000],dtype=n.complex64)
ZTI=n.zeros([2,2,6,nt,nrg],dtype=n.complex64)
n_blocks=int((b[1]-b[0])/L)

for bi in range(rank,n_blocks,size):
    b0=bi*L + b[0]
    RTI[:,:]=0.0
    Z[:,:]=0.0    
    for ippi in range(nt):
 #       print(ippi)
        i0=bi*L + ippi*1000 + b[0]
        z0=d.read_vector_c81d(i0,1000,"ch000")
#        plt.plot(z0.real)
 #       plt.plot(z0.imag)
  #      plt.show()
        z1=d.read_vector_c81d(i0,1000,"ch001")
        Z[0,ippi,:]=z0
        Z[1,ippi,:]=z1
    
    # groud clutter cancel
    if True:
        for ri in range(1000):
            Z[0,:,ri]=Z[0,:,ri]-n.mean(Z[0,:,ri])
            Z[1,:,ri]=Z[1,:,ri]-n.mean(Z[1,:,ri])#,n.repeat(1/100,100),mode="same")#-#n.mean(Z[1,:,ri])
        
#    plt.pcolormesh(Z[0,:,:].T.real)
#    plt.show()

    # deconvolve
    for ippi in range(nt):
#        print(ippi)        
        for i in range(len(sc.kb_code_seeds)):
            ZTI[0,0,i,ippi,:]=n.dot(jr[i]["B"],Z[0,ippi,:])
            ZTI[0,1,i,ippi,:]=n.dot(jr[i]["B"],Z[1,ippi,:])
            ZTI[1,0,i,ippi,:]=n.dot(kb[i]["B"],Z[0,ippi,:])
            ZTI[1,1,i,ippi,:]=n.dot(kb[i]["B"],Z[1,ippi,:])

    if True:
        P=n.zeros([2,ZTI.shape[3],ZTI.shape[4]])
        for i in range(len(sc.kb_code_seeds)):
            P+=n.abs(ZTI[:,0,i,:,:])**2.0
            P+=n.abs(ZTI[:,1,i,:,:])**2.0             
        plt.pcolormesh(P[0,:,:].T,cmap="plasma")
        plt.colorbar()
        plt.show()
        plt.pcolormesh(P[1,:,:].T,cmap="plasma")
        plt.colorbar()
        plt.show()
            
    wf=scipy.signal.windows.hann(nt)
    freqs=n.fft.fftshift(n.fft.fftfreq(nt,d=1000/100e3))
    rvec=n.arange(nrg)*3
    for ri in range(nrg):
        for i in range(len(sc.kb_code_seeds)):
            RTI[0,:,ri]+=n.fft.fftshift(n.abs(n.fft.fft(wf*ZTI[1,0,i,:,ri]))**2.0)
            RTI[0,:,ri]+=n.fft.fftshift(n.abs(n.fft.fft(wf*ZTI[1,1,i,:,ri]))**2.0)
 #   for fi in range(nt):
#        RTI[fi,:]=RTI[fi,:]/n.mean(RTI[fi,:])
    plt.pcolormesh(freqs,rvec,RTI[0,:,:].T)#,vmin=0)
    plt.xlabel("Doppler (Hz)")
    plt.ylabel("TX-RX range (km)")
#    plt.show()
            #        RTI[ippi,:]=RTI[ippi,:]/n.median(RTI[ippi,:])
#    for ri in range(1000):
        
#    plt.pcolormesh(10.0*n.log10(RTI.T),vmin=0)
    plt.title(stuffr.unix2datestr(b0/100e3))
    plt.colorbar()
    plt.tight_layout()
#    plt.show()
    plt.savefig("rti-%d.png"%(b0))
    print("rti-%d.png"%(b0))
    ho=h5py.File("rti-%d.h5"%(b0),"w")
    ho["RTI"]=RTI
    ho["t0"]=b0/100e3
    ho["rvec"]=rvec
    ho["freqs"]=freqs
    ho.close()
    plt.close()
#    plt.show()
        
