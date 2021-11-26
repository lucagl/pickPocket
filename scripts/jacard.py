#TODO:
# Definire misura di similarita <-- 2 (articolo Sala) OK
# Estrarre da due pockets residui (non atomi) --> usa roba gia pronta di pickPocket.. <-- 3 OK
# limita numero output pockets con opzione (lato pickPocket..)<-- 1
# file output similarita tra ogni coppia di pocket con sorting score di similarita OK

# OPEN QUESTIONS:
# merging of overlapping pockets in multiple frames?:
#    for instance p1,p9 --> p1_new for 1bid(holo) 3tms(apo)
# and for multiple frames? (trajectories..)
import re
import glob
import sys
import numpy as np

USE_RES=True
JACCARD_THRESHOLD = 0.1
frameFolder1='holo/1bid/'
frameFolder2='apo/3tms/'

def get_pAtoms(filename):
    '''
    Reads res info from PQR (pickPocket) or PDB (Fpocket) files 
    '''
    resMap = []
    content ={}
    comment =['#', 'CRYST[0-9]?','HEADER']
    remark = ['REMARK','HETATM']
    termination = ['TER', 'END', '\n']
    skip = comment+remark+termination
    skip = '(?:% s)' % '|'.join(skip)

    # print("-- Loading PQR file --")
    try:
        inFile = open(filename,'r')
    except Exception:
            raise NameError("Cannot load PQR file")
    for line in inFile: 
        if(re.match(skip,line)): 
            pass 
        else:
            linegNOChain=re.match("(ATOM)\s*(\d+)\s*(\S+)\s+([A-Z]+)\s+(\-?\d+[A-Z]?)\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s+(\-?\d*\.?\d+)\s+(\d*\.?\d+)",line)
            linegChain = re.match("(ATOM)\s*(\d+)\s*(\S+)\s+([A-Z]+)\s+([\w0-9]+)\s*(\-?\d+[A-Z]?)\s+(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s*(\-?\d*\.?\d+)\s+(\-?\d*\.?\d+)\s+(\d*\.?\d+)",line)
            break
    # print(line.split())
    # print(lineg)

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
    chainInd = 4

    # print(list(map(float, lineg[coordInd:coordInd+3])))

    inFile.seek(0)
    for line in inFile: 
        if(re.match(skip,line)): 
            pass 
        else:
            #DEBUG..
                # lineg = re.match(matchPattern,line)
                # if(not lineg):
                #     print(line)
            
            lineg = re.match(matchPattern,line).groups()

            if(isChainID):
                # content = {'resName':lineg[nameInd],'resNum':lineg[resInd],'atomNumber':int(lineg[1]),'resAtom': lineg[atomInd],'resChain':lineg[chainInd],
                # 'charge':float(lineg[chargeInd]),'coord':list(map(float, lineg[coordInd:coordInd+3])),'radius':float(lineg[rInd])}
                content = {'resName':lineg[nameInd],'resNum':lineg[resInd],'atomNumber':int(lineg[1]),'resAtom': lineg[atomInd],'resChain':lineg[chainInd],
                'coord':list(map(float, lineg[coordInd:coordInd+3]))}
            else:
                content = {'resName':lineg[nameInd],'resNum':lineg[resInd],'atomNumber':int(lineg[1]),'resAtom': lineg[atomInd],
                'coord':list(map(float, lineg[coordInd:coordInd+3]))}
            resMap.append(content)
            # print(content)
    
    
    return resMap



def get_uniqueRes(atoms,isRes):
    '''
    Input : pqr line of the pocket
    '''
    content=set()
    out=[]
    if(isRes):
        for a in atoms:
            resid =a ['resNum']#< MOST IMPORTANT..
            rname = a['resName']
            rChain = a['resChain']
            if(((resid,rname,rChain) in content)):
                pass
            else:
                out.append([resid,rname,rChain])
                content.add((resid,rname,rChain))
    else:
        for a in atoms:
            resid =a ['resNum']
            rname = a['resName']
            rChain = a['resChain']
            atomName = a['resAtom']
            out.append([resid,rname,rChain,atomName])
    return out

#### Principal function

def jack(resReference,resPocket,getMatchScores=False):
    '''
    Compares residue returning the Jaccard index
    J = |A intersection B| / |A union B |
    '''
  
    set_reference=set([tuple(l) for l in resReference])
    set_pocket=set([tuple(l) for l in resPocket])

    #Cardinality of the union:

    
    intersectionSet = set_pocket & set_reference
    I = len(intersectionSet)
    #DEBUG
    # unionSet = setA | setB
    # U = len(unionSet)
    #print('check U cardinality:', U, (len(setA)+len(setB) - I))
    # print('union = ',unionSet)
    # print('-------')
    # print('intersection = ', intersectionSet)


    U =( len(set_reference)+len(set_pocket) - I)
    # print(I,U)
    J = I/U
    OS = len(intersectionSet)/len(set_reference)
    VS = len(intersectionSet)/len(set_pocket)
    # J = np.round(J,4)

    if(getMatchScores):
        return J,OS,VS
    else:   
        return J






##########


def main():
    # folder1 = frameFolder1
    # folder2= frameFolder2
    
    pList_frame1=[n for n in glob.glob(frameFolder1+'p*_atm.pqr')] #not ranked   
    # print("unsorted",pList_frame1)
    pList_frame1 = sorted(pList_frame1,key=lambda x: int(re.sub('\D', '', x)))#ranked sorting

    pList_frame2=[n for n in glob.glob(frameFolder2+'p*_atm.pqr')] #not ranked   
    # print("unsorted",pList_frame2)
    pList_frame2 = sorted(pList_frame2,key=lambda x: int(re.sub('\D', '', x)))#ranked sorting
    print('FRAME 1:')
    print(pList_frame1)
    print('FRAME 2:')
    print(pList_frame2)

    print('Checking %d combinations'%(len(pList_frame1)*len(pList_frame2)))

    # CHECKS
    # a1 = get_pAtoms(pList_frame1[1])
    # print(pList_frame1[1])
    # res1 = get_uniqueRes(a1,USE_RES)
    # a2 = get_pAtoms(pList_frame2[4])
    # print(pList_frame2[4])
    # res2 = get_uniqueRes(a2,USE_RES)
    # print(len(res1))
    # print(res1)
    # print('------')
    # print(len(res2))
    # print(res2)
    # print("quantitative comparison")
    # res=jack(res1,res2)
    # print("%.2f%%"%(res*100))
    # # print(np.round(res,1))
    # input() 


    out = []
    samePocket = set()
    for r1,p1 in enumerate(pList_frame1):
        # print(p1)
        a1 = get_pAtoms(p1)
        res1 = get_uniqueRes(a1,USE_RES)
        d ={}
        for r2,p2 in enumerate(pList_frame2):
            # print(p2)
            a2 = get_pAtoms(p2)
            res2 = get_uniqueRes(a2,USE_RES)
            simScore = jack(res1,res2)
            if(np.round(simScore,1)>=JACCARD_THRESHOLD):
                d[r2] = simScore
                samePocket.add(r2)
        out.append(d)
            
            
    # FILE WITH ALL SIMILARITIES
    outfile=open('similarityMAP.txt','w')
    outfile.write('FRAME1\tFRAME2\n')
    for i in range(len(pList_frame1)):
        # outfile=open('p'+str(i+1)+'_similarity.txt','w')
        #sort according to similarity
        sortedSimilarity = sorted(out[i].items(), key=lambda kv: kv[1],reverse=True)
        outfile.write('p'+str(i+1)+'\n')
        for k in sortedSimilarity:
            outfile.write('\tp'+str(k[0]+1)+'\t score: '+str(k[1]*100)+'%%\n')

    outfile.close()
    #FILE FOR POCKETS IN FRAME 2 WITH LOW OVERLAP ("new pockets in frame2")
    outfile=open('newPockets.txt','w')
    for i in range(len(pList_frame2)):
        if i in samePocket:
            pass
        else:
            outfile.write('p'+str(i+1)+'\n')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser exit")
        sys.exit()