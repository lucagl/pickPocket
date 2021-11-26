
import re
import numpy as np
from  C_functs import Pdist_C,getIndex


def getStructureIterator(mapFile):
        
    comment = ['^#','\n']
    comment = '(?:% s)' % '|'.join(comment)
    # mapFile = 'ligandMap.txt'
    structures=[]
    try:
        inFile = open(mapFile,'r')
    except Exception:
        raise NameError("Cannot load mapFile")

    content = inFile.readlines()

    structures = []
    s=0
    while s <(len(content)):
        current_line=content[s]
        if(re.match(comment,current_line)):
            s+=1
            continue
        current_line = current_line.split()
        n_ligands = int(current_line[0])
        name = current_line[1]
        ligand_names=[]
        for i in range(1,n_ligands+1):
            following_line=content[s+i].split()[0]
            if(re.match(comment,following_line)):
                #skipping ligand
                continue
            ligand_names.append(following_line)
        structures.append({'pdb':name,'ligands': ligand_names})
        s+=n_ligands +1
            
    return structures

def get_protein(structure):
    '''
    Input: PQR file
    Output: List of dictionary. Each entry is a line in the PQR file.
    '''
    try:
        # print(structure+'.pdb')
        inFile = open(structure+'.pqr','r')
    except Exception:
        raise NameError("Cannot load PQR file")
    # try:
    #     # print(structure+'.pdb')
    #     _check = open(structure+'.pdb','r')
    # except Exception:
    #     raise NameError("Cannot load PDB file")
    comment =['#', 'CRYST[0-9]?']
    remark = ['REMARK']
    termination = ['TER', 'END', '\n']
    skip = comment+remark+termination
    skip = '(?:% s)' % '|'.join(skip)
    for line in inFile: 
        if(re.match(skip,line)): 
            pass 
        else:
            linegNOChain=re.match("(ATOM)\s*(\d+)\s*(\S+)\s+([A-Z]+)\s+(\-?\d+[A-Z]?)\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s+(\-?\d*\.?\d+)\s+(\d*\.?\d+)",line)
            linegChain = re.match("(ATOM)\s*(\d+)\s*(\S+)\s+([A-Z]+)\s+([\w0-9]+)\s*(\-?\d+[A-Z]?)\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s+(\-?\d*\.?\d+)\s+(\d*\.?\d+)",line)
            break

    if(linegChain):
        # print("PQR contains CHAIN_ID")
        isChainID=1                                                        #resID
        matchPattern = "(ATOM)\s*(\d+)\s*(\S+)\s+([A-Z]+)\s+([\w0-9]+)\s*(\-?\d+[A-Z]?)\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s+(\-?\d*\.?\d+)\s+(\d*\.?\d+)"
    elif(linegNOChain):
        # print("PQR does NOT contain CHAIN_ID")
        isChainID =0                                        # resID
        matchPattern = "(ATOM)\s*(\d+)\s*(\S+)\s+([A-Z]+)\s+(\-?\d+[A-Z]?)\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s+(\-?\d*\.?\d+)\s+(\d*\.?\d+)"
    else:
        raise NameError("Incorrect pqr file formatting")
    if(isChainID):
        resInd = 5
        chargeInd = 9
        rInd = 10
    else:
        resInd = 4
        chargeInd = 8
        rInd = 9
    nameInd = 3
    atomInd = 2
    coordInd = resInd +1
    chainInd=4
    
    inFile.seek(0)

    resMap=[]
    for line in inFile:
        if(re.match(skip,line)): 
            pass 
        else: 
            mline=re.match(matchPattern,line).groups()
            content = {'resName':mline[nameInd],'resNum':mline[resInd],'atomNumber':int(mline[1]),'resAtom': mline[atomInd],'resChain':mline[chainInd],
            'charge':float(mline[chargeInd]),'coord':list(map(float, mline[coordInd:coordInd+3])),'radius':float(mline[rInd])}
            resMap.append(content)
    
    return resMap

def readLigand(lname):
    '''
    Return ligand atoms read from lname
    '''
    inFile = open(lname+".xyz",'r')
    ligand_coord = np.loadtxt(inFile)

    return ligand_coord

def getCloseList(ligCoord,pCoord,thresholdDistance = 5):
    """
    Input ligand and protein coordinates
    Output indexes of protein coordinates closer than thresholdDistance to ligand
    """
    d,flag = Pdist_C(ligCoord,pCoord)
    index = np.where(d<=thresholdDistance)[0]
    _lindex,pindex=getIndex(flag,index,pCoord.shape[0])

    resList = np.unique(pindex)

    return resList

def ligPRes(ligname,resMap,savingDir,readingDir):
    """
    Produces a file containing the subset of protein pqr file close to the ligand
    """
    # List of protein atoms which are close to the ligand
    ligandCoord = readLigand(readingDir+ligname)
    proteinCoords = np.array([p['coord'] for p in resMap])
    resList=getCloseList(ligandCoord,proteinCoords)


    nameP = savingDir+"/"+ligname
    filename = nameP+"_atm.pqr"
    outFileP=open(filename,'w')

  
    for r in resList:
        outFileP.write("{:<6s}{:>5d} {:<5s}{:>3s} {:1s}{:>5s}   {:>8.3f} {:>8.3f} {:>8.3f} {:>8.4f} {:>8.4f}\n".format('ATOM',r+1,resMap[r]['resAtom'],resMap[r]['resName'],resMap[r]['resChain'],resMap[r]['resNum'],
        resMap[r]['coord'][0],resMap[r]['coord'][1],resMap[r]['coord'][2],resMap[r]['charge'],resMap[r]['radius']))
        # outFileP.write("ATOM  %5d%5s%4s%2s%4s    %8.3f%8.3f%8.3f%8.4f%8.4f\n" % (r+1,resMap[r]['resAtom'],resMap[r]['resName'],resMap[r]['resChain'],resMap[r]['resNum'],
        # resMap[r]['coord'][0],resMap[r]['coord'][1],resMap[r]['coord'][2],resMap[r]['charge'],resMap[r]['radius']))
    
    outFileP.close()






    ############################################# MAIN #############################################

def main():
    import os

    readFolder = 'structures/'
    outFolder = 'results/'


    # %%%%%%%%%%%%%%%% READ LP MAP %%%%%%%%%%%%%%%%%%%%%%%%%5
    structures = getStructureIterator(readFolder+'ligandMap.txt')
    print(structures)
    n_structures = len(structures)
    print("N structures to analyse:", n_structures)
    input()
    #   %%%%%%%%% LOOP THROUGH STRUCTURES %%%%%%%%%%%%
    for s in range(n_structures):
        proteinName = structures[s]['pdb']
        ligands_name = structures[s]['ligands']
        print(proteinName)
        print(str(len(ligands_name)) + " ligands:")
        # print(ligands_name)
        outName = outFolder+proteinName
        if not os.path.exists(outName):
            os.makedirs(outName)

        protein = get_protein(readFolder+proteinName)
    #   %%%%%%% LOOP THROUGH LIGANDS %%%%%%%%%%%%%%%%%
        for ln in ligands_name:
            print(ln)
            ligPRes(ln,protein,outName,readFolder)
        print('-----')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser exit")
        exit()