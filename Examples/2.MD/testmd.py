from ase.calculators.vasp.vaspFCP import VaspFCP
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import time
from ase.calculators.vasp import Vasp
from ase.io import read, write
from ase.optimize import LBFGS, BFGS, GPMin
from ase.optimize.bfgslinesearch import BFGSLineSearch
from ase.md import MDLogger
from ase.md.npt import NPT
from ase import units
import numpy as np
import os
import pandas as pd
from ase.constraints import FixAtoms,FixedPlane,FixBondLength
from ase.db import connect
from ase.calculators.plumed import Plumed

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
NELECT=222.3660124127793
#c= FixAtoms(indices=[8])
#atoms.set_constraint(c)
symb=str(U)
calculatorFP.set(label=symb, directory=symb,U=U,fpmethod ='Newton-fitting', NELECT=NELECT,NELECT0=NELECT0)

atoms.set_calculator(calculatorFP)


timestep = 0.5* units.fs
ps = 1000 * units.fs


###if perform MD
dyn = NPT(atoms, timestep, temperature_K=300, ttime=5 * units.fs,mask=(0,0,0),externalstress=0,trajectory='MTD.traj',append_trajectory=True)
dyn.attach(MDLogger(dyn, atoms, 'metadynamics.log', header=True, stress=False, peratom=False, mode="a"), interval=1)
dyn.run(20000)

'''
###if use PLUMED to perform contraint MD 
isexist=os.path.exists('MTD.traj')
if not isexist:
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    setup = [f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
             "e: ENERGY",
             "DISTANCE ATOMS=51,53 LABEL=d1",
             "DISTANCE ATOMS=52,53 LABEL=d2",
             "UPPER_WALLS ARG=d1 AT=3.2 KAPPA=500.",
             "UPPER_WALLS ARG=d2 AT=3.2 KAPPA=500."          #constraint the distances of atoms 
             "PRINT ARG=* FILE=COLVAR"
             #+ " GRID_RFILE=GRID"
             ]   
    atoms.calc = Plumed(calc=calculatorFP,
                        input=setup,
                        timestep=timestep,
                        atoms=atoms,
                        #restart=True,
                        log='MTD.txt',
                        kT=300*units.kB)
    #atoms.calc.istep = 50
    dyn = NPT(atoms, timestep, temperature_K=300, ttime=5 * units.fs,mask=(0,0,0),externalstress=0,trajectory='MTD.traj',append_trajectory=True)
    dyn.attach(MDLogger(dyn, atoms, 'metadynamics.log', header=True, stress=False, peratom=False, mode="a"), interval=1)
    dyn.run(20000)
else:
    configurations = read('MTD.traj',index=':')
    istep=len(configurations)
    atoms.set_positions(configurations[-1].get_positions())
    atoms.set_momenta(configurations[-1].get_momenta())
    NELECT=get_number_of_electrons(file='-1.5/OUTCAR')
    calculatorFP.set(NELECT=NELECT)
    setup = [f"UNITS LENGTH=A TIME={1/ps} ENERGY={units.mol/units.kJ}",
             "e: ENERGY",
             "DISTANCE ATOMS=51,53 LABEL=d1",
             "DISTANCE ATOMS=52,53 LABEL=d2",
             "UPPER_WALLS ARG=d1 AT=3.2 KAPPA=500.",
             "UPPER_WALLS ARG=d2 AT=3.2 KAPPA=500.",
             "PRINT ARG=* FILE=COLVAR"
             #+ " GRID_RFILE=GRID"
             ]   
    atoms.calc = Plumed(calc=calculatorFP,
                        input=setup,
                        timestep=timestep,
                        atoms=atoms,
                        restart=True,
                        log='MTD.txt',
                        kT=300*units.kB)
    atoms.calc.istep = istep
    dyn = NPT(atoms, timestep, temperature_K=300, ttime=5 * units.fs,mask=(0,0,0),externalstress=0,trajectory='MTD.traj',append_trajectory=True)
    dyn.attach(MDLogger(dyn, atoms, 'metadynamics.log', header=True, stress=False, peratom=False, mode="a"), interval=1)
    dyn.run(20000)
'''
