from ase.calculators.vasp.vaspFCP import VaspFCP
import time
from ase.calculators.vasp import Vasp
from ase.io import read, write
from ase.optimize import LBFGS, BFGS, GPMin
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.constraints import FixAtoms,FixedPlane,FixBondLength


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

 
#Set up VASP calculator        ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
calculator=Vasp(xc='PBE', #functional
          pp='PBE',            #type of pseudopotential
          kpts=(2, 2, 1),      #kpoint
          ncore=4,
          ispin=2,lasph=True,ismear=0, sigma=0.1, algo='Fast', ediff=1E-5, prec='Accurate',  encut=400,  nelm=500 , addgrid='Ture',lreal='Auto',lorbit=11, ivdw=11, #parameters for SCF
          #tau=0, lrhoion=False, lsol=True, eb_k=78.4, lambda_d_k=3.0, #parameters for vaspsol
          lwave=True,              #write WAVECAR to speed up the SCF of the next ionic step
          )
calculator.set(label='vacuum', directory='vacuum')


#Set up VASP calculator        ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
calculatorFP=VaspFCP(xc='PBE', #functional
          pp='PBE',            #type of pseudopotential
          kpts=(2, 2, 1),      #kpoint
          ncore=4,
          ispin=2,lasph=True,ismear=0, sigma=0.1, algo='Fast', ediff=1E-5, prec='Accurate',  encut=400,  nelm=500 , addgrid='Ture',lreal='Auto',lorbit=11, ivdw=11, #parameters for SCF
          tau=0, lrhoion=False, lsol=True, eb_k=78.4, lambda_d_k=3.0, #parameters for vaspsol
          lwave=True,              #write WAVECAR to speed up the SCF of the next ionic step
          )
calculatorFP.set(label='relax', directory='relax')


U=-1.5
atoms=read('POSCAR')


for atom in atoms:
    if atom.symbol=='Fe':
        atom.magmom=0.001

NELECT0=222
#or NELECT0=get_number_of_electrons(file='./vacuum/OUTCAR')
NELECT=222

#c= FixAtoms(indices=[8])
#atoms.set_constraint(c) ###fix atoms

symb=str(U)
calculatorFP.set(label=symb, directory=symb,U=U,fpmethod ='Newton-fitting', NELECT=NELECT,NELECT0=NELECT0)
atoms.set_calculator(calculatorFP)
dyn=LBFGS(atoms)
dyn.run(fmax=0.01)
