from ase.calculators.FCPelectrochem import FCP

from ase.calculators.vasp import Vasp
from ase.io import read,
from ase.optimize import LBFGS




##########################################################
##########################################################
########################################################
#########################################################

def get_number_of_electrons(file='OUTCAR'):  
    #read NELECT of PZC from the OUTCAR of a vacuum calculation
    with open(file, 'r') as fd:
        for line in fd.readlines():
            if 'total number of electrons' in line:
                nelect = float(line.split('=')[1].split()[0].strip())
                break
    return nelect

 

cal_sol=Vasp(xc='PBE', #functional
          pp='PBE',            #type of pseudopotential
          kpts=(3, 3, 1),      #kpoint
          ncore=4,
          ispin=2,lasph=True,ismear=0, sigma=0.1, algo='Fast', ediff=1E-5, prec='Accurate',  encut=400,  nelm=500 , addgrid='Ture',lreal='Auto',lorbit=11, ldau_luj={'Fe': {'L': 2, 'U': 5.0, 'J': 0.46}}, lmaxmix=4, #parameters for SCF
          tau=0, lrhoion=False, lsol=True, eb_k=78.4, lambda_d_k=3.0, #parameters for vaspsol
          lwave=True, lcharg = False,              #write WAVECAR to speed up the SCF of the next ionic step
          )
cal_sol.set(label='sol', directory='sol')


cal_FP=FCP(innercalc=cal_sol,
                 fcptxt='log-fcp.txt',
                 U=0.8,
                 NELECT = 208.5,
                 C = 1/80,    #1/k  capacitance per A^2
                 FCPmethod = 'Newton-fitting',
                 FCPconv=0.01,
                 NELECT0=210, 
                 adaptive_lr=False,
                 work_ref=4.6,
                 max_FCP_iter=10000
          )


#________________________________________________________________

atoms=read('POSCAR')
atoms.calc=cal_FP
dyn=LBFGS(atoms, trajectory='fp.traj')
dyn.run(fmax=0.01)










