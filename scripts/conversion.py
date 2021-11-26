#QUESTION: What about subpockets?

import pandas as pd
import re
import numpy as np
import sys


SES_RADIUS = 1.8

OS_Threshold = 0.5
VS_Threshold = 0.2

conversion = pd.read_pickle("../conversion.pickle")
def holo2apo(holoid: str, chain: str, resnumb: str) -> str:
    """Converts the resnumb of a HOLO structure
    into its corresponding resnumb in the APO structure.

    Args:
        holoid (str): HOLO structure pdb code.
        chain (str): chain of the residue to be converted.
        resnumb (str): residue number of the residue to be converted.
          Append the insertion code if present i.e. "60A"

    Returns:
        str: correspoding residue number in the APO structure.
          Insertion code is append when present.
    """
    try:
        resnumb=int(resnumb)
    except ValueError:
        # print('resnumb with letter..')
        pass

    prot_conversion = conversion.query(
        f"HOLOid == '{holoid}' and HOLOchain == '{chain}'"
    )
    i_res = prot_conversion["HOLOresnumbs"].values[0].index(resnumb)
    converted_resnumb = prot_conversion["APOresnumbs"].values[0][i_res]
    apo_chain = prot_conversion["APOchain"].values[0]


    return str(converted_resnumb),apo_chain
##############

def convertResNum(oldResidues,holoNAME):
    newResidues = oldResidues
    for i,r in enumerate(oldResidues):
        resNUM = r[0]
        resCHAIN = r[2]
        newNUM,newCHAIN = holo2apo(holoNAME,resCHAIN,resNUM)
        newResidues[i][0] = newNUM
        newResidues[i][2] = newCHAIN
    return newResidues




def ReadHOLOMap(filename):
    map ={}
    f = open(filename,'r')
    for line in f:
        # print(line)
        match = re.search('(\S+)\s+(\S+)',line)
        if match:
            apo=match.group(1)
            holo = match.group(2)
            map[apo] = holo
        else:
            print('no match..')
            print(line)
    return map


mapping = ReadHOLOMap('../correspondenceMAP.txt')
print("Map=",mapping)
####################### NS INTERFACE functions
from pickPocket.functions import setup_NSInput,new_probe,fetchRes
import subprocess


class exposedFetcher(object):
    def __init__(self) -> None:
        self.conf = 'surfaceConfiguration.prm'
        self.workingDir = 'temp/'
        self.indices=set()
        # self.resMap =[]
    def getExposedAtoms(self,structure_name):
        print('settung up configuration file NS')
        setup_NSInput(self.workingDir+self.conf,accTriang=False)
        new_probe(self.workingDir+self.conf,SES_RADIUS)
        print('setting up structure file %s for NS'%structure_name)
        resMap = fetchRes('../apoPQRs/'+structure_name+'.pqr') #also produces NS input file by default
        protein_atoms = np.empty((0,4))
        for i in resMap:
            c = np.append(np.asarray(i['coord']),i['radius'])
            protein_atoms = np.vstack([protein_atoms,c])
        np.savetxt(self.workingDir+"NanoShaper_input.xyzr",protein_atoms,delimiter="\t",fmt='%.4f')
        try:
            #call NS
            out = subprocess.check_output('./NanoShaper',cwd=self.workingDir)
        except subprocess.CalledProcessError as grepexc:                                                                                                   
            print ("error code", grepexc.returncode, grepexc.output)

        #CHECK EXPOSED
        exposedFile = open(self.workingDir+'exposedIndices.txt','r')
        for line in exposedFile.readlines():
            self.indices.add(int(line))
        print('Before exposing:', len(resMap))
        self.map=set([(d['resNum'],d['resChain']) for d in list(filter(lambda x: x['atomNumber'] in self.indices, resMap))])
        print('After exposing:', len(self.map))
    def checkExposed(self,atoms):
        #important: pass only residues of interest (the converted apo ones), check if those residues have good exposed indices    
        #x[0] = resNumber, x[2] = resChain (resname useless actually)
        residuesFiltered = list(filter(lambda x : (x[0],x[2]) in self.map, atoms))#[d for d in atoms if d['atomNumber'] in self.indices]
        return residuesFiltered
 

####
# New score like holo
# def matchScore(resPocket,resReference):
#     set_pocket=set([tuple(l) for l in resPocket])
#     set_reference=set([tuple(l) for l in resReference])
#     intersectionSet = set_pocket & set_reference

#     OS = len(intersectionSet)/len(set_reference)
#     VS = len(intersectionSet)/len(set_pocket)
#     return OS,VS

def main():
    print('ASSESSMENT OF APO POCKETS. NORM IS LIGAND-STRUCTURE NORM')
    import jacard
    import glob
    USE_RES=True
    JACCARD_THRESHOLD = 0.1
    norm =0 
    hitTOP10=0
    hitTOP3=0
    avScoreJC =0
    avScoreOS =0
    avScoreVS =0
    VS_low=0
    exposed = exposedFetcher()
    print('VS_threshold=',VS_Threshold,'OS_threshold=',OS_Threshold)
    print('SES radius exposed:', SES_RADIUS)
    input('\n')


    #Prepara mappa di tutte le pocket tra loro con jackart non nullo (file a parte di testo).
    #e unisci scores pockets che si toccano ? 
    pocketVSlow=[]
    for holo,apo in mapping.items():
        print(holo,apo)
        apoFolder = apo+'_pockets/'
        pList_apo=[n for n in glob.glob(apoFolder+'p*_atm.pqr')] #not ranked   
        pList_apo = sorted(pList_apo,key=lambda x: int(re.sub('\D', '', x)))#ranked sorting
        # print("sorted pockets",pList_apo)
        refPocketFolder = '../referenceHOLO/'+holo+'/'
        groundTruth =  [n for n in glob.glob(refPocketFolder+'*')] #can have more than 1 ligand
        # print("ground truth files:",groundTruth)
        exposed.getExposedAtoms(apo)
        for refP in groundTruth:
            atomRef = jacard.get_pAtoms(refP)
            #FILTER HERE FOR EXPOSED..


            ###
            resReference = jacard.get_uniqueRes(atomRef,USE_RES)
            # print("before conversion res Reference pocket:", resReference)
            resReference = convertResNum(resReference,holo) 
            # print("after conversion (%d elements):%s"%(len(resReference),resReference))  
            resReference=exposed.checkExposed(resReference)
            # print("After exposed filtering (%d elements):%s"%(len(resReference),resReference))
            norm+=1
            for r,pAPO in enumerate(pList_apo):
                # print(pAPO)
                # print('rank=',r+1)
                atomAPO = jacard.get_pAtoms(pAPO)
                resAPO = jacard.get_uniqueRes(atomAPO,USE_RES)  
                
                simScore,OS,VS = jacard.jack(resReference,resAPO,getMatchScores=True)
                # print('OS=',OS,'VS=',VS)
                # print('JACCARD:',simScore)
                #if(np.round(simScore,1)>=JACCARD_THRESHOLD):
                    
                if(((np.round(OS,1)>=OS_Threshold) and (np.round(VS,1) >= VS_Threshold))):
                    print('**HIT:%s'%(pAPO))
                    print('OS=',OS,'VS=',VS,'JACCARD:',simScore)
                    hitTOP10+=1
                    avScoreJC += simScore
                    avScoreOS += OS
                    avScoreVS += VS
                    if(r<3):
                        print('TOP3 COUNT')
                        hitTOP3+=1
                    if(np.round(VS,1)<0.5):
                        print('Scenario where VS low')
                        VS_low+=1
                        pocketVSlow.append(pAPO)
                        # input()
                    break
                if(OS>=0.1):
                    print('%s Almost'%(pAPO))
                    print('OS=',OS,'VS=',VS,'JACCARD:',simScore)
                    # print(resAPO)
                    # input()
        # input()
    # hitTOP10 = hitTOP10/norm
    # hitTOP3 = hitTOP3/norm
    # avScore = avScore/norm

    print('hitTOP10=%.2f\thitTOP3=%.2f\tJACCARD Score=%.2f\tOVERLAP Score=%.2f\tVOLUME Score=%.2f'%(np.round((hitTOP10/norm),2),np.round((hitTOP3/norm),2),
    np.round((100*avScoreJC/norm),2),np.round((100*avScoreOS/norm),2),np.round((100*avScoreVS/norm),2)))
    print('NORM=',norm)
    print('VS low %d times'%VS_low)
    print(pocketVSlow)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser exit")
        sys.exit()