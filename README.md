# FCP-vasp-ase
ASE interface for fully constant potential simulations with the Vienna Ab initio Simulation Package (VASP)

version 1.0


This module is modified based on the ASE interface to VASP.

The original interface was developed on the basis of modules by Jussi Enkovaara and John Kitchin.  
The constant-potential version was developed by Zhaoming Xia.



## Before you use

1. If you want to run fully constant potential calculation properly, the vaspsol code (https://github.com/henniggroup/VASPsol) should be included in the source code directory of VASP before compiling VASP; the patch of vaspsol should be applied to compute the FERMI_SHIFT; add -Dsol_compat option to the list of precompiler options(CPP_OPTIONS) in the makefile of VASP, then compile VASP.

2. Make sure you have installed python and pip.

3. Install ase by 'pip install ase'.

2. Copy vaspFCP.py to "python_lib_path/ase/calculators/vasp/".

3. If you want to use PLUMED interface, read this page (https://wiki.fysik.dtu.dk/ase/ase/calculators/plumed.html).

## How to use

1. The path of the directory containing the pseudopotential directories (potpaw,potpaw_GGA, potpaw_PBE, ...) should be set by the environmental flag $VASP_PP_PATH.

2. A simple example to call the module:

'''

    from ase.calculators.vasp.vaspFCP import VaspFCP
    from ase.io import read
    
    calculator=VaspFCP(xc='PBE', #functional
          pp='PBE',            #type of pseudopotential
          kpts=(2, 2, 1),      #kpoint
          ncore=4,
          ispin=2,lasph=True,ismear=0, sigma=0.1, algo='Fast', ediff=1E-5, prec='Accurate',  encut=400,  nelm=500 , addgrid='Ture',lreal='Auto',lorbit=11, ivdw=11,     #parameters for SCF
          tau=0, lrhoion=False, lsol=True, eb_k=78.4, lambda_d_k=3.0, #parameters for vaspsol
          lwave=True,              #write WAVECAR to speed up the SCF of the next ionic step
          )

    atoms=read('POSCAR')
    calculator.set(U=-1.5, NELECT=222.3660124127793,NELECT0=222)
    atoms.set_calculator(calculator)

'''

3. VaspFCP is compatible with all the parameters of Vasp. And it has some extra parameters:

'''

            U: float
               The potential of working electrode (V vs. reference electrode).
               defult:None
            
            NELECT: float
                initial guass of number of electrons.
                defult:None
            
            NELECT0: float
                number of electrons of the potential of zero charge (PZC). 
                defult:None

            FCPmethod: str
                method to run constant-potential calculation:
                'Newton-fitting'   (recommanded)
                'scipyBFGS'
                'scipyCG'
                'scipyLBFGS'
                defult:'Newton-fitting'

            FCPconv: float
                 converagence of delta_miu (eV) for constant-potential calculation. 
                 defult:0.01

            work_ref:float
                the work function (the negative value of absolut electrode potential) of reference electrode.
                If the reference electrode is SHE, work_ref is set to the defult value. (4.6)
                If the reference electrode is RHE, you should set the work_ref manually according the pH.
                defult:4.6
       
            C:float
                initial guass of capacitance per surface area. (e/V/(Å^2))
                defult:1/80
'''

