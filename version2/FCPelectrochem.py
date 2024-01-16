import os
import time
from numpy.polynomial.chebyshev import Chebyshev
fit=Chebyshev.fit
import pandas as pd
import subprocess
from warnings import warn
from typing import Dict, Any
import numpy as np
from ase.calculators import calculator
from ase.calculators.calculator import Calculator,FileIOCalculator
from ase.calculators.vasp import Vasp
from ase.io import write
from ase.io.vasp import write_vasp
from ase.parallel import world

class FCP(FileIOCalculator):
    """Fully constant potential calculator for electrochemistry"""

    name='FCP'
    implemented_properties = [
        'energy', 'free_energy', 'forces', 'dipole', 'fermi', 'stress','magmom', 'magmoms'
    ]

     # Can be used later to set some ASE defaults
    default_parameters: Dict[str, Any] = {}

    def __init__(self,
                 atoms=None,
                 innercalc=Vasp,
                 fcptxt='log-fcp.txt',
                 U=None,
                 NELECT = None,
                 C = 1/80,    #1/k  capacitance per A^2
                 FCPmethod = 'Newton-fitting',
                 FCPconv=0.01,
                 NELECT0=None, 
                 adaptive_lr=False,
                 work_ref=4.6,
                 max_FCP_iter=10000,
                 always_adjust=True,
                 explicit_sol=False,
                 **kwargs):
        '''
        always_adjust
          defult: True
          Adjust ne again even when potential is within tolerance. This is useful to set to True along with a loose potential tolerance (FCPconv) to allow the potential and structure to be simultaneously optimized in a geometry optimization, for example. Default: False.
        '''
        
        FileIOCalculator.__init__(self)
        
        
        self._atoms = None
        self.results = {}

        self.workSHE=work_ref
        self.wf= U + work_ref
        self.FCPconv=FCPconv
        self.Nelect=NELECT
        self.Nelect0=NELECT0
        self.Cpersurf=C
        self.always_adjust=always_adjust
        self.explicit_sol=explicit_sol
        

        self.FCPmethod=FCPmethod
        self.max_FCP_iter=max_FCP_iter
        self.adaptive_lr=adaptive_lr
        self.fcptxt=fcptxt
        self.innercalc=innercalc
        with open(self.fcptxt, mode='w',encoding='utf-8') as f:
            f.write('loop'+'\t'+'NELECT'+'\t' +'Fermi(eV)'+'\t'+'Fermishift(eV)'+'\t'+'mu(eV)'+'\t' + 'U(V)' + '\t' + 'conv(V)'+'\t'+'Ewithoutentropy(eV)'+'\t'+'Ewithoutentropy_grand(eV)' +'\t'+'Etoten(eV)'+'\t'+'Etoten_grand(eV)'+'\t'+ 'Cpersurf(e/V/A^2)'+'\n')



    def set(self, **kwargs):
        changed_parameters = {}

        if 'label' in kwargs:
            self.label = kwargs.pop('label')

        if 'fcptxt' in kwargs:
            self.fcptxt = kwargs.pop('fcptxt')

        if 'innercalc' in kwargs:
            self.innercalc = kwargs.pop('innercalc')

        if 'always_adjust' in kwargs:
            self.always_adjust = kwargs.pop('always_adjust')

        if 'explicit_sol' in kwargs:
            self.explicit_sol=kwargs.pop('explicit_sol')

        if 'atoms' in kwargs:
            atoms = kwargs.pop('atoms')
            self.atoms = atoms  # Resets results

        if 'U' in kwargs:
            self.U = kwargs.pop('U')

        if 'NELECT' in kwargs:
            self.Nelect = float(kwargs.pop('NELECT'))

        if 'NELECT0' in kwargs:
            self.Nelect0 = float(kwargs.pop('NELECT0'))

        if 'C' in kwargs:
            self.Cpersurf = float(kwargs.pop('C')) 
            

        if 'FCPmethod' in kwargs:
            self.FCPmethod = kwargs.pop('FCPmethod')

        if 'adaptive_lr' in kwargs:
            self.adaptive_lr=kwargs.pop('adaptive_lr')

        if 'FCPconv' in kwargs:
            self.FCPconv=float(kwargs.pop('FCPconv'))

        if 'work_ref' in kwargs:
            self.workSHE=float(kwargs.pop('work_ref'))

        if 'max_FCP_iter' in kwargs:
            self.max_FCP_iter=int(kwargs.pop('max_FCP_iter'))
            
        changed_parameters.update(FileIOCalculator.set(self, **kwargs))
        if changed_parameters:
            self.clear_results()  # We don't want to clear atoms

    def reset(self):
        self.atoms = None
        self.clear_results()

    def clear_results(self):
        self.results.clear()
        

    def calculate(self, atoms=None,properties=('energy', ),
                  system_changes=tuple(calculator.all_changes)):


        self.clear_results()
        if atoms is not None:
            self.atoms = atoms.copy()

        self.C=self.Cpersurf * np.linalg.norm(np.cross(self.atoms.cell[0],self.atoms.cell[1]))

        LogPath=self.innercalc.directory

        if not os.path.exists(LogPath):
            os.mkdir(LogPath)

        with open(LogPath+ '/tmp-log-FCP.txt', mode='w',encoding='utf-8') as f:
            f.write('loop'+'\t'+'NELECT'+'\t' +'Fermi(eV)'+'\t'+'Fermishift(eV)'+'\t'+'mu(eV)'+'\t' + 'Ucal(V)' + '\t' + 'conv(V)'+'\t'+'Ewithoutentropy(eV)'+'\t'+'Ewithoutentropy_grand(eV)' +'\t'+'Etoten(eV)'+'\t'+'Etoten_grand(eV)'+'\t'+ 'Cpersurf(e/V/A^2)'+'\t'+ 'time(s)'+'\n')
        
        if self.wf == None:
            raise calculator.CalculationFailed('please set U (vs. reference electrode)')

        def Cevalue(Ne):
            data=pd.read_csv(LogPath+'/tmp-log-FCP.txt',sep='\t')
            if len(data['NELECT'])>=2:
                pfit,fitdis=fit(data['NELECT'].values,data['Fermi(eV)'].values+data['Fermishift(eV)'].values,deg=1,full=True)
                if len(data['NELECT'])>3 and fitdis[0][0]>0.1:
                    pfit,fitdis=fit(data['NELECT'].values,data['Fermi(eV)'].values+data['Fermishift(eV)'].values,deg=2,full=True)
                    if len(data['NELECT'])>4 and fitdis[0][0]>0.1:
                        pfit,fitdis=fit(data['NELECT'].values,data['Fermi(eV)'].values+data['Fermishift(eV)'].values,deg=3,full=True)

                K=(pfit(Ne+0.0001)-pfit(Ne))/0.0001 
                if K<=0:
                    os.system('echo dfermi/dne is not positive'  + ' >> ' + self.directory + '/WARNING.txt')
                else:
                    self.C=1/K
                    self.Cpersurf=self.C/np.linalg.norm(np.cross(self.atoms.cell[0],self.atoms.cell[1]))
                os.system('echo '+ str(1/self.Cpersurf) + ' >> '+ self.directory +'/K.txt')
            
            #os.system('echo '+ str(self.C)+ ' >>  C.txt')
        
        self.FCPloop=0
        lr=1.0      # learning rate 0<lr<=1
        lrcount=0
        convold=None
        Nelectold=None
        while self.FCPloop < self.max_FCP_iter:
            self.FCPloop += 1
            startcal=time.time()
            atomstmp=self.atoms.copy()
            
            if self.innercalc.name=='vasp':
                self.innercalc.set(nelect=self.Nelect)
            else:
                raise calculator.CalculationFailed('the calculator is not supported yet')

            atomstmp.calc=self.innercalc
            
            energy_free = atomstmp.get_potential_energy(force_consistent=True)
            energy=atomstmp.get_potential_energy(force_consistent=False)
            if self.innercalc.name=='vasp':
                self.fermishift=self.read_fermishift_vaspsol(outpath=atomstmp.calc.directory+'/'+atomstmp.calc.txt)
            #elif  user-defined Fermishift 
            else:
                raise calculator.CalculationFailed('the calculator is not supported yet')
            #print(atomstmp.calc.results)
            self.fermi=atomstmp.calc.get_fermi_level()
            
            Ucal=-(self.fermi + self.fermishift)-self.workSHE
            conv=self.wf + self.fermi + self.fermishift
            grand_energy_free=energy_free+(self.wf+self.fermishift)*(self.Nelect-self.Nelect0)
            grand_energy=energy+(self.wf+self.fermishift)*(self.Nelect-self.Nelect0)
            endcal=time.time()
            with open(LogPath+ '/tmp-log-FCP.txt', mode='a',encoding='utf-8') as f:
                print("%d\t%11.6f\t%11.6f\t%11.6f\t%7.3f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%7.0f" %(self.FCPloop, self.Nelect, self.fermi, self.fermishift, -self.wf, Ucal, conv, energy, grand_energy,energy_free,grand_energy_free,self.Cpersurf, endcal-startcal), file = f)

            Cevalue(self.Nelect)
            if convold != None:
                if convold*conv < 0:
                    lrcount+=1
            convold = conv
            Nelectold=self.Nelect
            if self.always_adjust == False:
                if abs(conv)<self.FCPconv:
                    break

    
            if self.FCPmethod == 'Newton-fitting':
                if self.adaptive_lr==True:
                    if lrcount >1:
                        lr=lr-0.5**lrcount
                    #print(str(Nelectold),str(lr),str(convold),str(self.C))
                self.Nelect=Nelectold-lr*convold*self.C  #lr is learning rate

            if abs(self.Nelect-Nelectold)/self.Nelect > 0.05: 
                os.system('rm '+self.directory +'/WAVECAR')

            magmoms=atomstmp.calc.results['magmoms']
            #print(magmoms)
            self.atoms.magmoms=list(magmoms)

            if abs(conv)<self.FCPconv:
                break

        if self.explicit_sol=True:
            '''
            The coexistence of implicit solvent and explicit solvent will lead to the double counting of solvent effect. Thus, implicit solvent should be removed after the Nelect is converged.
            '''
            if self.innercalc.name=='vasp':
                self.innercalc.set(lsol=False,ldipol=True,idipol=3)
            else:
                raise calculator.CalculationFailed('the calculator is not supported yet')

            atomstmp.calc=self.innercalc
            energy_free = atomstmp.get_potential_energy(force_consistent=True)
            energy=atomstmp.get_potential_energy(force_consistent=False)
            grand_energy_free=energy_free+(self.wf+self.fermishift)*(self.Nelect-self.Nelect0)
            grand_energy=energy+(self.wf+self.fermishift)*(self.Nelect-self.Nelect0)
            


        self.results.update(
        dict(magmom=atomstmp.calc.results['magmom'], 
            magmoms=atomstmp.calc.results['magmoms'], 
            dipole=atomstmp.calc.results['dipole'], 
            nbands=atomstmp.calc.results['nbands'],
            energy=grand_energy,
            free_energy=grand_energy_free, 
            forces=atomstmp.calc.results['forces'], 
            #fermi=self.fermi, 
            stress=atomstmp.calc.results['stress'],
            ))
        

        with open(self.fcptxt, mode='a',encoding='utf-8') as f:
            print("%d\t%11.6f\t%11.6f\t%11.6f\t%7.3f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f\t%11.6f" %(self.FCPloop, self.Nelect, self.fermi, self.fermishift, -self.wf, Ucal, conv, energy, grand_energy,energy_free,grand_energy_free,self.Cpersurf), file = f)


    def read_fermishift_vaspsol(self,outpath, lines=None):
        """Method that reads Fermi energy from vasp.out file"""
        if not lines:
            lines = self.load_file(outpath)

        E_fs = None
        for line in lines:
            if 'FERMI_SHIFT' in line:
                if float(line.split()[2]) !=0.0:
                    E_fs = float(line.split()[2])
        return E_fs
    
    def load_file(self, filename):
        """Reads a file in the directory, and returns the lines

        Example:
        >>> outcar = load_file('OUTCAR')
        """
        with open(filename, 'r') as fd:
            return fd.readlines()


