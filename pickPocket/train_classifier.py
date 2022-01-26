#TODO efficiency:
# Can we avoid vomputing subrank for all pockets but focus only on subpockets when needed ?

import numpy as np
from pickPocket import global_module
from pickPocket.global_module import Error
from pickPocket.functions import Pdist_C,getIndex,getEntrance
import re



rmin=global_module.rmin

ML = {"IsolationForest":1,"Extended IsolationForest":2}

minsize = int(global_module.count_Threshold/10)
# minsize =10
# shiftThreshold=4
shiftThreshold = global_module.shift_Threshold


# Hard coding classes (20 Canonical aminoacids)
hClass = {}
hClass['ILE'] = 1 #Very hydrophobic (very low polarity)
hClass['VAL'] = 1 
hClass['LEU'] = 2 #Hydrophobic
hClass['ALA'] = 3 #Quite hydrophobic
hClass['CYS'] = 3
hClass['PHE'] = 3
hClass['MET'] = 3
hClass['GLY'] = 4 #~neutral
hClass['PRO'] = 5 #hydrophilic
hClass['SER'] = 5
hClass['THR'] = 5
hClass['TRP'] = 5
hClass['TYR'] = 5 
hClass['ASP'] = 6 #Very hidrophilic
hClass['GLU'] = 6
hClass['HIS'] = 6

#Alternative nomenclature for HIS
hClass['HID'] = 6
hClass['HIE'] = 6
hClass['HIP'] = 6
#########

hClass['LYS'] = 6 
hClass['ASN'] = 6
hClass['GLN'] = 6 
hClass['ARG'] = 6 

likeWater={}
likeWater[1] = -3
likeWater[2] = -2
likeWater[3] = -1
likeWater[4] = 0
likeWater[5] = 1
likeWater[6] = 2


resClass = {}
resClass['ALA'] = 1
resClass['ARG'] = 2 
resClass['ASN'] = 3
resClass['ASP'] = 4 
resClass['CYS'] = 5
resClass['GLN'] = 6
resClass['GLU'] = 7
resClass['GLY'] = 8 
resClass['HIS'] = 9
#Alternative nomenclature for HIS
resClass['HID'] = 9
resClass['HIE'] = 9
resClass['HIP'] = 9
#########
resClass['ILE'] = 10 
resClass['LEU'] = 11
resClass['LYS'] = 12 
resClass['MET'] = 13
resClass['PHE'] = 14
resClass['PRO'] = 15 
resClass['SER'] = 16
resClass['THR'] = 17
resClass['TRP'] = 18
resClass['TYR'] = 19
resClass['VAL'] = 20

def loadDict(fileName):
    import pickle
    dictList = []
    with open(fileName, 'rb') as dbfile:
        while True:
            try:
                db = pickle.load(dbfile)
            except EOFError:
                break
            dictList.append(db)
    return dictList

def concatenateRanks(a,b):
    ''' Concatenate and then like np unique but preserving order'''
    c = np.concatenate([a,b])
    ind = np.unique(c, return_index=True)[1]
    return np.array([c[index] for index in sorted(ind)])

def orderPreservedUnique(a):
    u,ind = np.unique(a,return_index = True)
    return u[np.argsort(ind)]

def weightSamples(weigth_list,original_samples,isP=False):
    #1. RENORMALIZE WEIGHTS
    VS_min = 0.2
    OS_min = 0.5
    if(isP):
        print("Large pockets weights")
        minscore = VS_min
    else:
        print("Small pockets weights")
        minscore =0.5*(VS_min + OS_min)
    
    weight_list = np.array(weigth_list)
    weight_list = (weight_list-minscore)/(1-minscore)

    print(weight_list[:100])
    #2. REWEIGHTING
    import random
    if (isinstance(original_samples, np.ndarray)):
        original_samples=original_samples.tolist()
    return_samples=original_samples.copy()
    for k,s in enumerate(original_samples):
        r = random.random()
        if ((r<=weigth_list[k])):
            # return_samples = np.vstack((return_samples, s))
            return_samples +=[s]
        else:
            pass
    
    return np.array(return_samples)
                


def getRes(resMap,atomList,getCoord=False,getAtomNumber=False):
        """
        Returns residue info of given atoms list (parsed like atom indexes)
        atomList are checked for redundancies (expected, since the spheres can be tangent to identical atoms)
        NEW: mask where the residues where within 4 AA of ligand coord, if given.
        """

        dummyChain=None
        resList = np.asarray(atomList)
        resList = np.unique(resList)
        # resMap= self.protein.resMap
        if(getAtomNumber):
            resInfo=  [resMap[r]['atomNumber'] for r in resList]
            return resInfo 
        #check if chain provided 
        if (not('resChain' in resMap[0])):
            dummyChain = 'A'
            if(getCoord):
                resInfo = [(resMap[r]['resName'],resMap[r]['resNum'],dummyChain,tuple(resMap[r]['coord']+[resMap[r]['radius']])) for r in resList]
            else:
                resInfo = [(resMap[r]['resName'],resMap[r]['resNum'],dummyChain) for r in resList]
        else:
            if(getCoord):
                resInfo = [(resMap[r]['resName'],resMap[r]['resNum'],resMap[r]['resChain'],tuple(resMap[r]['coord']+[resMap[r]['radius']])) for r in resList]
                # resInfo = [(resMap[r]['resAtom'],resMap[r]['resName'],resMap[r]['resNum'],np.append(np.asarray(resMap[r]['coord']),resMap[r]['radius'])) for r in resList]
            else:
                resInfo = [(resMap[r]['resName'],resMap[r]['resNum'],resMap[r]['resChain']) for r in resList]
        
        return resInfo

def hydroCounter(res):
    countClass = np.zeros(6)
    # n=len(f['res'])
    n=0
    content = set()
    for r in res:
        # key = (r[0],r[1])
        resname = r[0]
        resid = r[1]
        resChain = r[2]
        # print(resid,resname)
        if (resid,resname,resChain) in content:
            continue
        content.add((resid,resname,resChain))

        if resname in hClass:
            countClass[hClass[resname]-1]+=1
            # print("found", resname, pClass[resname])
        
    norm = len(content)
    # print('Normalization =',norm)
    if norm ==0:
        normCount = countClass*0 #might happen with ligand contact since the trheshold was 4 angstrom instead than 5 (Walter's idea..)
    else:
        normCount = np.round(countClass/norm *100,2)
    
    return normCount # already as a percentage..

def resCounter(res):
    countClass = np.zeros(20)
    # n=len(f['res'])
    n=0
    content = set()
    for r in res:
        # key = (r[0],r[1])
        resname = r[0]
        resid = r[1]
        resChain = r[2]
        # print(resid,resname)
        if (resid,resname,resChain) in content:
            continue
        content.add((resid,resname,resChain))

        if resname in resClass:
            countClass[resClass[resname]-1]+=1
            # print("found", resname, pClass[resname])
        
    norm = len(content)
    # print('Normalization =',norm)
    if norm ==0:
        normCount = countClass*0 #might happen with ligand contact since the trheshold was 4 angstrom instead than 5 (Walter's idea..)
    else:
        normCount = np.round(countClass/norm *100,2)
    
    return normCount # already as a percentage..

def densityHydro(res):
    
    D = 5
    
    
    N = len(res)
    # print(N)
    # density = len(res)/f['volume']
    # norm=N*np.pi*D*D#number of atoms X circular shell (I,m considering surface atoms..)
   
    
    # More senseful but bad limit.. Tends to decrease if D larger than effective radius area of pocket 
    # (facing atoms, which we don't know precisely)
    # entrR = np.array([e[1] for e in f['entrances_rBased']])
    # # print(entrR)
    # areaEntrances = sum(entrR**2 * np.pi)
    # area = abs(f['area']-areaEntrances)
    # norm=(np.pi*D*D)*(N**2)/area # = N (averaged atoms) V(r) (spherical shell) N/V density long range
    # print(f['area'],areaEntrances,area)


    # norm = N*N


    coord = np.array([r[3] for r in res])

    d,_f=Pdist_C(coord[:,0:3],coord[:,0:3])
    index = np.where(d<D)[0]
    # print(index)
    # indexesMAP = []
    counter = np.ones(N)
    # counterHP = np.zeros(N)#Hydrophobic
    # counterHPL = np.zeros(N)#Hydrophilic
    counterHPL = np.array([1 if likeWater[hClass[r[1]]]>0 else 0 for r in res]) #Hydrophilic
    counterHP = np.array([1 if likeWater[hClass[r[1]]]<0 else 0 for r in res]) #Hydrophobic
    # print(counterHP)
    # print(counterHPL)
    for k in index:
        # indexesMAP.append(getIndex(1,k,norm)) #CHECK
        resnameA = res[getIndex(1,k,N)[0]][1]
        resnameB = res[getIndex(1,k,N)[1]][1]
        if((likeWater[hClass[resnameA]]>0)and(likeWater[hClass[resnameB]]>0)):
            counterHPL[getIndex(1,k,N)[0]]+=1
            counterHPL[getIndex(1,k,N)[1]]+=1
        elif((likeWater[hClass[resnameA]]<0)and(likeWater[hClass[resnameB]]<0)):
            counterHP[getIndex(1,k,N)[0]]+=1
            counterHP[getIndex(1,k,N)[1]]+=1
       
        counter[getIndex(1,k,N)[0]]+=1
        counter[getIndex(1,k,N)[1]]+=1

    #Like this is the actual density in a sphere of D=5. Is more a surrogate of g(r) with no need to define area
    #If GLY are present is less dense but this is ok.
    # REMARK: is important in this sense to use same D as the one used to define matching scores.

    density_hydrophilic = sum(counterHPL)/sum(counter)*100 
    density_hydrophobic = sum(counterHP)/sum(counter) *100
    return density_hydrophilic,density_hydrophobic






def buildTestFeatures(pocket,resMap,imSub=False):
    """
    Return the feature list of a single pocket.
    IMPORTANT: Respect the ordering of the training features
    """

    err = Error()

    # +++++++++++++ CHEMICAL FEATURES +++++++++++++++
    
    resData = getRes(resMap,pocket['node'].getAtomsIds(),getCoord=True)
    # res = [(r[2],r[3],r[4]) for r in resData]
    countHydroClasses = hydroCounter(resData)
    hydrophilic_score = sum(countHydroClasses[4:6])
    hydrophobic_score = sum(countHydroClasses[0:3])
    countRes = resCounter(resData)
    densityHydrophilic,densityHydrophobic = densityHydro(resData) 
    #could be redundant with hydroScores above (this is conceptually better though)


    # +++++++++++++ GEOMETRICAL/CLUSTERING FEATURES ++++++++++++++++++

    nn = pocket['node'].count #NORMALIZATION FACTOR
    
    volume,area,err = pocket['node'].NSVolume(accurate=True)
    if(err.value==1):
        #Trying with ARVO
        volume,area,err = pocket['node'].volume()
        print("Using ARVO to compute volume..")
        if(err.value==1):
            return err
    
    HW = np.pi**(1./3.)*(6*volume)**(2./3.)/area
    depthList = np.asarray([d for d,r,count in pocket['persistence']])
    aV = [a[1:] for a in pocket['aggregations']] #aggregation vector
    aVr = list(filter(lambda x:(max(x[2],x[3])/min(x[2],x[3]) <= 5) and min(x[2],x[3]) >=minsize, aV))#FILTERED AGGREGATIONS FOR RELEVANT EVENTS 
    bV = pocket['btlnk_info']

    isCavity = 0
    if(imSub):
        #Thus is a "subpocket" and I have to hand build this raw feature
        clusteredEntrances = [(pocket['depth'],pocket['rtop'],pocket['rtop'],1)]
        entrances = [(pocket['node'].coord,pocket['depth'],pocket['rtop'],pocket['rtop'],1)]
        n_entrances = 1
        if(clusteredEntrances[0][1]<rmin):
            # print(clusteredEntrances[0][1])
            isCavity=1
    else:
        entrances=getEntrance(pocket['large_mouths'],pwDist=True)
        clusteredEntrances = [e[1:] for e in entrances]
        n_entrances = len(clusteredEntrances)
    
        if(n_entrances == 0):
            isCavity = 1 #abuse of notation, could also be a deep cleft..
            entrances=getEntrance(pocket['mouths'],pwDist=True)  
            clusteredEntrances = [e[1:] for e in entrances]
            n_entrances = len(clusteredEntrances)


    pocket['entrances'] = entrances #Save entrances in the dictionary --> useful for debugging

    # +++ Entrance scores +++
    
    av_radius_entrances = sum([r_eff for d,r_eff,r_av,cluster_size in clusteredEntrances])/n_entrances
    #I think it make sense to weight this with the entrance cluster score..
    av_entrance_depth = sum([d*cluster_size for d,r_eff,r_av,cluster_size in clusteredEntrances])/sum([cluster_size for d,r_eff,r_av,cluster_size in clusteredEntrances])
    
    # +++++++++++ geometrical scores ++++++++
    ram =  np.std(depthList)
    tr = abs(av_entrance_depth - np.average(depthList)) - ram #"Protrusion" =  ramification on superficial features
    av_entrance_depth = av_entrance_depth/shiftThreshold # 1 is the minimum

    # +++++ aggregation and clustering  Scores +++++
    score = (sum(depthList)+len(aV))/nn *100
    aggr_score = len(aV)/nn *100
    relevantAggr_score = len(aVr)/nn *100
    #COMMENT: so far I'm not using radii
    pers_score=sum(depthList)/nn *100 #how many aligned sferes over total 

    if(aVr):
        average_relevantAggrRadius = sum([(a[2]+a[3])*a[1] for a in aVr])/sum([a[2]+a[3] for a in aVr]) #weigth more important aggregations
    else:
        average_relevantAggrRadius = 0
    # ++++ bottleneck scores +++
    n_bottl = len(bV)
    if(bV):
        average_r_bottl = sum(r for r,n in bV)/n_bottl
    else:
        average_r_bottl=0
    
    n_bottl = len(bV)/nn * 100
  
    Xg=[nn,volume,HW,n_bottl,average_r_bottl,pers_score,aggr_score,relevantAggr_score,average_relevantAggrRadius,
    n_entrances,av_radius_entrances,av_entrance_depth,score,volume/nn, tr,ram, isCavity]

    Xc=[countRes[0],countRes[1],countRes[2],countRes[3],countRes[4],countRes[5],countRes[6],countRes[7],countRes[8],
        countRes[9],countRes[10],countRes[11],countRes[12],countRes[13],countRes[14],countRes[15],countRes[16],countRes[17],countRes[18],
        countRes[19],hydrophilic_score, hydrophobic_score,densityHydrophilic,densityHydrophobic]
   

    return err,Xg,Xc

def buildGeomFeatures(pocket,imSub=False):
    """
    Return the feature list of a single pocket.
    IMPORTANT: Respect the ordering of the training features
    """
    

    err = Error()


    # +++++++++++++ GEOMETRICAL/CLUSTERING FEATURES ++++++++++++++++++

    nn = pocket['node'].count #NORMALIZATION FACTOR
    
    volume,area,err = pocket['node'].NSVolume(accurate=True)
    if(err.value==1):
        #Trying with ARVO
        volume,area,err = pocket['node'].volume()
        print("Using ARVO to compute volume..")
        if(err.value==1):
            return err
    
    HW = np.pi**(1./3.)*(6*volume)**(2./3.)/area
    depthList = np.asarray([d for d,r,count in pocket['persistence']])
    aV = [a[1:] for a in pocket['aggregations']] #aggregation vector
    aVr = list(filter(lambda x:(max(x[2],x[3])/min(x[2],x[3]) <= 5) and min(x[2],x[3]) >=minsize, aV))#FILTERED AGGREGATIONS FOR RELEVANT EVENTS 
    bV = pocket['btlnk_info']

    isCavity = 0
    if(imSub):
        #Thus is a "subpocket" and I have to hand build this raw feature
        clusteredEntrances = [(pocket['depth'],pocket['rtop'],pocket['rtop'],1)]
        entrances = [(pocket['node'].coord,pocket['depth'],pocket['rtop'],pocket['rtop'],1)]
        n_entrances = 1
        if(clusteredEntrances[0][1]<rmin):
            # print(clusteredEntrances[0][1])
            isCavity=1
    else:
        entrances=getEntrance(pocket['large_mouths'],pwDist=True)
        clusteredEntrances = [e[1:] for e in entrances]
        n_entrances = len(clusteredEntrances)
    
        if(n_entrances == 0):
            isCavity = 1 #abuse of notation, could also be a deep cleft..
            entrances=getEntrance(pocket['mouths'],pwDist=True)  
            clusteredEntrances = [e[1:] for e in entrances]
            n_entrances = len(clusteredEntrances)


    pocket['entrances'] = entrances #Save entrances in the dictionary --> useful for debugging

    # +++ Entrance scores +++
    
    av_radius_entrances = sum([r_eff for d,r_eff,r_av,cluster_size in clusteredEntrances])/n_entrances
    #I think it make sense to weight this with the entrance cluster score..
    av_entrance_depth = sum([d*cluster_size for d,r_eff,r_av,cluster_size in clusteredEntrances])/sum([cluster_size for d,r_eff,r_av,cluster_size in clusteredEntrances])
    
    # +++++++++++ geometrical scores ++++++++
    ram =  np.std(depthList)
    tr = abs(av_entrance_depth - np.average(depthList)) - ram #"Protrusion" =  ramification on superficial features
    av_entrance_depth = av_entrance_depth/shiftThreshold # 1 is the minimum

    # +++++ aggregation and clustering  Scores +++++
    score = (sum(depthList)+len(aV))/nn *100
    aggr_score = len(aV)/nn *100
    relevantAggr_score = len(aVr)/nn *100
    #COMMENT: so far I'm not using radii
    pers_score=sum(depthList)/nn *100 #how many aligned sferes over total 

    if(aVr):
        average_relevantAggrRadius = sum([(a[2]+a[3])*a[1] for a in aVr])/sum([a[2]+a[3] for a in aVr]) #weigth more important aggregations
    else:
        average_relevantAggrRadius = 0
    # ++++ bottleneck scores +++
    n_bottl = len(bV)
    if(bV):
        average_r_bottl = sum(r for r,n in bV)/n_bottl
    else:
        average_r_bottl=0
    
    n_bottl = len(bV)/nn * 100
  
    Xg=[nn,volume,HW,n_bottl,average_r_bottl,pers_score,aggr_score,relevantAggr_score,average_relevantAggrRadius,
    n_entrances,av_radius_entrances,av_entrance_depth,score,volume/nn, tr,ram, isCavity]

   
    return err,Xg

def buildChemFeaturesNoDensity(pocket,resMap):
    """
    Return the feature list of a single pocket.
    IMPORTANT: Respect the ordering of the training features
    """
    

    err = Error()

    # +++++++++++++ CHEMICAL FEATURES +++++++++++++++
    
    resData = getRes(resMap,pocket['node'].getAtomsIds())
    # res = [(r[2],r[3],r[4]) for r in resData]
    countHydroClasses = hydroCounter(resData)
    hydrophilic_score = sum(countHydroClasses[4:6])
    hydrophobic_score = sum(countHydroClasses[0:3])
    countRes = resCounter(resData)
  

    Xc=[countRes[0],countRes[1],countRes[2],countRes[3],countRes[4],countRes[5],countRes[6],countRes[7],countRes[8],
        countRes[9],countRes[10],countRes[11],countRes[12],countRes[13],countRes[14],countRes[15],countRes[16],countRes[17],countRes[18],
        countRes[19],hydrophilic_score, hydrophobic_score]
   

    return err,Xc


class CNNFeature(object):
    def __init__(self):
        super().__init__() #class inheritance method.. useless there's no inheritance here
        self.pege = __import__('pege')
        
    def init(self,structureName):
        err=Error()
        # print('loading structure')
        try:
            self.pegeProtein = self.pege.Pege(structureName+'.pqr')
        except Exception:
            err.value=2
            err.info = "Cannot load structure for PEGE calculations.."
        return err
    def build(self,pocket,resMap):
        
        err = Error()
        p_atm = getRes(resMap,pocket['node'].getAtomsIds(),getAtomNumber=True)
        
        pfeat = self.pegeProtein.get_atoms(p_atm,ignore_missing=True)
        if(not isinstance(pfeat,type(None))):
            pfeat=pfeat.tolist()
        else:
            err.value = 1
        
        return err,pfeat        



def buildChemFeaturesDensity(pocket,resMap):
    """
    Return the feature list of a single pocket.
    IMPORTANT: Respect the ordering of the training features
    """
    

    err = Error()

    # +++++++++++++ CHEMICAL FEATURES +++++++++++++++
    
    resData = getRes(resMap,pocket['node'].getAtomsIds(),getCoord=True)
    # res = [(r[2],r[3],r[4]) for r in resData]
    densityHydrophilic,densityHydrophobic = densityHydro(resData)
    countRes = resCounter(resData)
    

    Xc=[countRes[0],countRes[1],countRes[2],countRes[3],countRes[4],countRes[5],countRes[6],countRes[7],countRes[8],
        countRes[9],countRes[10],countRes[11],countRes[12],countRes[13],countRes[14],countRes[15],countRes[16],countRes[17],countRes[18],
        countRes[19],densityHydrophilic, densityHydrophobic]
   

    return err,Xc



###### USELESSS #############3
def buildTestFeaturesUnique(pocket,resMap,imSub=False):
    """
    Return the feature list of a single pocket.
    IMPORTANT: Respect the ordering of the training features
    """

    err = Error()

    # +++++++++++++ CHEMICAL FEATURES +++++++++++++++
    
    resData = getRes(resMap,pocket['node'].getAtomsIds(),getCoord=True)
    # res = [(r[2],r[3],r[4]) for r in resData]
    countHydroClasses = hydroCounter(resData)
    hydrophilic_score = sum(countHydroClasses[4:6])
    hydrophobic_score = sum(countHydroClasses[0:3])
    countRes = resCounter(resData)
    densityHydrophilic,densityHydrophobic = densityHydro(resData) 
    #could be redundant with hydroScores above (this is conceptually better though)


    # +++++++++++++ GEOMETRICAL/CLUSTERING FEATURES ++++++++++++++++++

    nn = pocket['node'].count #NORMALIZATION FACTOR
    
    volume,area,err = pocket['node'].NSVolume(accurate=True)
    if(err.value==1):
        #Trying with ARVO
        volume,area,err = pocket['node'].volume()
        print("Using ARVO to compute volume..")
        if(err.value==1):
            return err
    
    HW = np.pi**(1./3.)*(6*volume)**(2./3.)/area
    depthList = np.asarray([d for d,r,count in pocket['persistence']])
    aV = [a[1:] for a in pocket['aggregations']] #aggregation vector
    aVr = list(filter(lambda x:(max(x[2],x[3])/min(x[2],x[3]) <= 5) and min(x[2],x[3]) >=minsize, aV))#FILTERED AGGREGATIONS FOR RELEVANT EVENTS 
    bV = pocket['btlnk_info']

    isCavity = 0
    if(imSub):
        #Thus is a "subpocket" and I have to hand build this raw feature
        clusteredEntrances = [(pocket['depth'],pocket['rtop'],pocket['rtop'],1)]
        entrances = [(pocket['node'].coord,pocket['depth'],pocket['rtop'],pocket['rtop'],1)]
        n_entrances = 1
        if(clusteredEntrances[0][1]<rmin):
            # print(clusteredEntrances[0][1])
            isCavity=1
    else:
        entrances=getEntrance(pocket['large_mouths'],pwDist=True)
        clusteredEntrances = [e[1:] for e in entrances]
        n_entrances = len(clusteredEntrances)
    
        if(n_entrances == 0):
            isCavity = 1 #abuse of notation, could also be a deep cleft..
            entrances=getEntrance(pocket['mouths'],pwDist=True)  
            clusteredEntrances = [e[1:] for e in entrances]
            n_entrances = len(clusteredEntrances)


    pocket['entrances'] = entrances #Save entrances in the dictionary --> useful for debugging

    # +++ Entrance scores +++
    
    av_radius_entrances = sum([r_eff for d,r_eff,r_av,cluster_size in clusteredEntrances])/n_entrances
    #I think it make sense to weight this with the entrance cluster score..
    av_entrance_depth = sum([d*cluster_size for d,r_eff,r_av,cluster_size in clusteredEntrances])/sum([cluster_size for d,r_eff,r_av,cluster_size in clusteredEntrances])
    
    # +++++++++++ geometrical scores ++++++++
    ram =  np.std(depthList)
    tr = abs(av_entrance_depth - np.average(depthList)) - ram #"Protrusion" =  ramification on superficial features
    av_entrance_depth = av_entrance_depth/shiftThreshold # 1 is the minimum

    # +++++ aggregation and clustering  Scores +++++
    score = (sum(depthList)+len(aV))/nn *100
    aggr_score = len(aV)/nn *100
    relevantAggr_score = len(aVr)/nn *100
    #COMMENT: so far I'm not using radii
    pers_score=sum(depthList)/nn *100 #how many aligned sferes over total 

    if(aVr):
        average_relevantAggrRadius = sum([(a[2]+a[3])*a[1] for a in aVr])/sum([a[2]+a[3] for a in aVr]) #weigth more important aggregations
    else:
        average_relevantAggrRadius = 0
    # ++++ bottleneck scores +++
    n_bottl = len(bV)
    if(bV):
        average_r_bottl = sum(r for r,n in bV)/n_bottl
    else:
        average_r_bottl=0
    
    n_bottl = len(bV)/nn * 100
  
    Xt=[nn,volume,HW,n_bottl,average_r_bottl,pers_score,aggr_score,relevantAggr_score,average_relevantAggrRadius,
    n_entrances,av_radius_entrances,av_entrance_depth,score,volume/nn, tr,ram, isCavity,countRes[0],countRes[1],countRes[2],countRes[3],countRes[4],countRes[5],countRes[6],countRes[7],countRes[8],
        countRes[9],countRes[10],countRes[11],countRes[12],countRes[13],countRes[14],countRes[15],countRes[16],countRes[17],countRes[18],
        countRes[19],hydrophilic_score, hydrophobic_score,densityHydrophilic,densityHydrophobic]
   

    return err,Xt

##################################################

class MLmodel(object):
    """
    To be flexible in the integration of other ML methods
    """
    def __init__(self, externalObj, mType = 1):
        self.mType=mType
        self.myself = externalObj
        self.ntrees = None
        self.sample = None
        if(mType == ML['IsolationForest']):
            self.ntrees = self.myself.get_params(False)["n_estimators"]
            self.sample = self.myself.get_params(False)["max_samples"]
            self.bootstrap =self.myself.get_params(False)["bootstrap"]
            self.n_features = self.myself.n_features_
            self.exlevel = 0 #by definition from EIF perspective..
        elif(mType == ML['Extended IsolationForest']):
            self.ntrees = self.myself.ntrees
            self.sample = self.myself.sample
            self.bootstrap =True #I think..
            self.exlevel = self.myself.exlevel
    def getScore(self,Xt):
        # print(type(Xt))
        # print(len(Xt))
        Xt = np.array(Xt)
        # print(Xt)
        if(self.mType== ML['IsolationForest']):
            # STANDARD ISOLATION FOREST
            return -self.myself.score_samples(Xt)
        elif(self.mType==ML['Extended IsolationForest']):
            #EXTENDED ISOLATION FOREST
            return self.myself.compute_paths(Xt)


class Scoring(object):
    #Can be trained or loaded with an already trained model 
    def __init__(self,modelL=None,modelS=None,modelChem=None,modelChemS = None,modelP=None,modelPall=None):
        """
        Allows to load a pretrained model
        """
        self._modelL = modelL
        self._modelS = modelS
        self._modelChem= modelChem
        self._modelChemS= modelChemS

        self._modelP = modelP
        self._modelPall = modelPall

        self._Sscore=None
        self._Lscore = None
        

        self._chemScore = None
        self._chemScoreS = None

        self._pScore = None
        self._pScoreS = None

        # self._features = None
        #OLD..
        self._globalRank = None
        self._globalScore = None

        # self.TrainsamplesL = []
        # self.TrainsamplesS = []




    def load(self,filename_model,modelType,unique=False,useCNN=False):
        err = Error()
        import pickle
        filename_model = global_module.trainingData+filename_model
        print("Loading ML trained model")
        print(filename_model)
        print("Checking for large pocket and small pocket interpreters")
        if(unique):
            try:
                inFile=open(filename_model+"_L.pkl",'rb')
                self._modelL= MLmodel(pickle.load(inFile),modelType)
                inFile.close()
                inFile=open(filename_model+"_S.pkl",'rb')
                self._modelS= MLmodel(pickle.load(inFile),modelType)
                inFile.close()
            except:
                err.value=2
                err.info = "Cannot load trained model(s) in "+global_module.trainingData
                return err
            if(modelType==ML["IsolationForest"]):
                # Isolation Forest
                print("Standard Isolation Forest")
                nTrees = self._modelL.ntrees
                sample = self._modelL.sample
                bootstrap = self._modelL.bootstrap
                n_features = self._modelL.n_features
                if(sample=="auto"):
                    sample =256
                print("Unique tree with geometry and chemistry")
                print("Number of trees = %d, n features = %d, sample size = %d, bootstrap = %s"%(nTrees,n_features,sample,bootstrap))
        
        else:
            
            try:
                print("Loading geometry")
                inFile=open(filename_model+"_geometryL.pkl",'rb')
                self._modelL= MLmodel(pickle.load(inFile),modelType)
                inFile.close()
                inFile=open(filename_model+"_geometryS.pkl",'rb')
                self._modelS= MLmodel(pickle.load(inFile),modelType)
                inFile.close()
                
                if(useCNN==True):
                    print("Loading CNN interactions")
                    
                    inFile=open(filename_model+"_cnnL.pkl",'rb')
                    self._modelP= MLmodel(pickle.load(inFile),modelType)
                    inFile.close()
                    inFile=open(filename_model+"_cnnS.pkl",'rb')
                    self._modelPall= MLmodel(pickle.load(inFile),modelType)
                    inFile.close()
                #else:
                    # print("Using resnames and hydroscores as chemical features ")
                print("Loading chemistry")
                inFile=open(filename_model+"_chemistryL.pkl",'rb')
                self._modelChem= MLmodel(pickle.load(inFile),modelType)
                inFile.close()
                inFile=open(filename_model+"_chemistryS.pkl",'rb')
                self._modelChemALL= MLmodel(pickle.load(inFile),modelType)
                inFile.close()
            except:
                err.value=2
                err.info = "Cannot load trained model(s)"
                return err
                ### PRINT INFO ###

            if(modelType==ML["IsolationForest"]):
                # Isolation Forest
                print("Standard Isolation Forest")
                nTrees = self._modelL.ntrees
                sample = self._modelL.sample
                bootstrap = self._modelL.bootstrap
                n_features = self._modelL.n_features
                if(sample=="auto"):
                    sample =256
                print("Geometry")
                print("Number of trees = %d, n features = %d, sample size = %d, bootstrap = %s"%(nTrees,n_features,sample,bootstrap))

                nTrees = self._modelChem.ntrees
                sample = self._modelChem.sample
                bootstrap = self._modelChem.bootstrap
                n_features = self._modelChem.n_features
                if(sample=="auto"):
                    sample =256
                print("Chemistry")
                print("Number of trees = %d, n features = %d, sample size = %d, bootstrap = %s"%(nTrees,n_features,sample,bootstrap))
                if(useCNN==True):
                    nTrees = self._modelP.ntrees
                    sample = self._modelP.sample
                    bootstrap = self._modelP.bootstrap
                    n_features = self._modelP.n_features
                    print("CNN forest")
                    print("Number of trees = %d, n features = %d, sample size = %d, bootstrap = %s"%(nTrees,n_features,sample,bootstrap))
        if(modelType==ML["Extended IsolationForest"]):
            # Isolation Forestsample
            print("Extended Isolation Forest")
            nTrees = self._modelL.ntrees
            sample = self._modelL.sample
            exlevel = self._modelL.exlevel
            print("Number of trees = %d, sample size = %d, extension level= %d"%(nTrees,sample,exlevel))
        return err

        

    
    def _getScores(self,featListGeom=[],featListChem=[],featListPedro=[]):
        if(featListGeom):
            Lscore = self._modelL.getScore(featListGeom)
            Sscore = self._modelS.getScore(featListGeom)
            self._Lscore = Lscore
            self._Sscore=Sscore
        if(featListChem):
            
            chemScore = self._modelChem.getScore(featListChem)
            chemScoreS = self._modelChemALL.getScore(featListChem)
            self._chemScore = chemScore
            self._chemScoreS = chemScoreS
        if(featListPedro):
            Pscore = self._modelP.getScore(featListPedro)
            PscoreS = self._modelPall.getScore(featListPedro)
            self._pScore = Pscore
            self._pScoreS = PscoreS

        return 
    def getScoresUnique(self,featList):
        LscoreU = self._modelL.getScore(featList)
        SscoreU = self._modelS.getScore(featList)
        return LscoreU,SscoreU

    def resetRank(self):
        self._Lscore = None
        self._Sscore = None
        self._chemScore = None
        self._chemScoreS = None

        #TODO: OLD.. drop
        self._globalRank = None
        self._globalScore = None


    def getRanking(self,featListGeom,featListChem):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        
        chemScore = self._chemScore
        chemScoreS = self._chemScoreS
        avScoreL = 0.5 * (Lscore + chemScore)
        
        # Sscore = 0.5*(Lscore + Sscore) #Alternative inspired by previous analysis (globalRank) NO better S forest
        avScoreS = 0.5 * (Sscore + chemScoreS)
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,np.sort(np.round(avScoreL,3)),np.sort(np.round(avScoreS,3))

    def getRanking_noSorting(self,featListGeom,featListChem):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        
        chemScore = self._chemScore
        chemScoreS = self._chemScoreS
        avScoreL = 0.5 * (Lscore + chemScore)
        
        # Sscore = 0.5*(Lscore + Sscore) #Alternative inspired by previous analysis (globalRank) NO better S forest
        avScoreS = 0.5 * (Sscore + chemScoreS)
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,avScoreL,avScoreS

    def getRankingPedro(self,featListGeom,featListChem,featListPedro):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem,featListPedro)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        
        chemScore = self._chemScore
        chemScoreS = self._chemScoreS

        Pscore = self._pScore
        PscoreS = self._pScoreS

        avScoreL = 1./3. * (Lscore + chemScore + Pscore)
        
        # Sscore = 0.5*(Lscore + Sscore) #Alternative inspired by previous analysis (globalRank) NO better S forest
        avScoreS = 1./3. * (Sscore + chemScoreS + PscoreS)
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,np.sort(np.round(avScoreL,3)),np.sort(np.round(avScoreS,3))

    def getRankingALL(self,featListGeom,featListChem):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        # chemScore = self._chemScore
        chemScoreS = self._chemScoreS
        avScoreL = 0.5 * (Lscore + chemScoreS)
        avScoreS = 0.5 * (Sscore + chemScoreS)
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,np.sort(np.round(avScoreL,3)),np.sort(np.round(avScoreS,3))

    def getRanking8020(self,featListGeom,featListChem):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        chemScore = self._chemScore
        chemScoreS = self._chemScoreS

        avScoreL = 0.8*Lscore + 0.2*chemScore
        Sscore = 0.5*(Lscore + Sscore) #Alternative inspired by previous analysis (globalRank)
        avScoreS = 0.8*Sscore + 0.2*chemScoreS
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,np.sort(np.round(avScoreL,3)),np.sort(np.round(avScoreS,3))


    def getRanking8020ALL(self,featListGeom,featListChem):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        # chemScore = self._chemScore
        chemScoreS = self._chemScoreS

        avScoreL = 0.8*Lscore + 0.2*chemScoreS
        avScoreS = 0.8*Sscore + 0.2*chemScoreS
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,np.sort(np.round(avScoreL,3)),np.sort(np.round(avScoreS,3))

    def getRanking2080(self,featListGeom,featListChem):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        chemScore = self._chemScore
        chemScoreS = self._chemScoreS

        avScoreL = 0.2*Lscore + 0.8*chemScore
        avScoreS = 0.2*Sscore + 0.8*chemScoreS
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,np.sort(np.round(avScoreL,3)),np.sort(np.round(avScoreS,3))
    
    def getRanking2080ALL(self,featListGeom,featListChem):
            
        if self._Lscore is None:
                self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        # chemScore = self._chemScore
        chemScoreS = self._chemScoreS

        avScoreL = 0.2*Lscore + 0.8*chemScoreS
        avScoreS = 0.2*Sscore + 0.8*chemScoreS
        
        mainRank  = np.argsort(avScoreL)
        subrank = np.argsort(avScoreS)

        return mainRank,subrank,np.sort(np.round(avScoreL,3)),np.sort(np.round(avScoreS,3))
    
    def getRankingOnlyGeom(self,featListGeom):

        if self._Lscore is None:
            self._getScores(featListGeom)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        
        # Sscore = 0.5*(Lscore + Sscore) 

        mainRank  = np.argsort(Lscore)
        subrank = np.argsort(Sscore)

        return mainRank,subrank,np.sort(np.round(Lscore,3)),np.sort(np.round(Sscore,3))
    
    
    def getRankingOnlyChem(self,featListChem):
        
        if self._chemScore is None:
            self._getScores(featListChem=featListChem)
        else:
            pass
        chemScore = self._chemScore
        chemScoreS = self._chemScoreS
        mainRank  = np.argsort(chemScore)
        subrank = np.argsort(chemScoreS)

        return mainRank,subrank,np.sort(np.round(chemScore,3)),np.sort(np.round(chemScoreS,3))

    def getRankingOnlyPedro(self,featListPedro):
        
        if self._chemScore is None:
            self._getScores(featListPedro=featListPedro)
        else:
            pass
        Pscore = self._pScore
        PscoreS = self._pScoreS
        mainRank  = np.argsort(Pscore)
        subrank = np.argsort(PscoreS)

        return mainRank,subrank,np.sort(np.round(Pscore,3)),np.sort(np.round(PscoreS,3))
    
    def getRankingOnlyChemALL(self,featListChem):
        
        if self._chemScore is None:
            self._getScores(featListChem=featListChem)
        else:
            pass
        # chemScore = self._chemScore
        chemScoreS = self._chemScoreS
        mainRank  = np.argsort(chemScoreS)
        subrank = np.argsort(chemScoreS)

        return mainRank,subrank,np.sort(np.round(chemScoreS,3)),np.sort(np.round(chemScoreS,3))
    

    def getRankingUnique(self,featList):
        
        LscoreUnique,SscoreUnique = self.getScoresUnique(featList)
        avScore = 0.5*(LscoreUnique + SscoreUnique)
        mainRank  = np.argsort(LscoreUnique)
        subrank = np.argsort(avScore)

        return mainRank,subrank,np.sort(np.round(LscoreUnique,3)),np.sort(np.round(avScore,3))

    def getRankingUniqueAverage(self,featList):
        
        LscoreUnique,SscoreUnique = self.getScoresUnique(featList)
        avScore = 0.5*(LscoreUnique + SscoreUnique)
        mainRank  = np.argsort(avScore)
        subrank = np.argsort(SscoreUnique)

        return mainRank,subrank,np.sort(np.round(avScore,3)),np.sort(np.round(SscoreUnique,3))
    
    def specialRanking1(self,featListGeom,featListChem):
        '''In out mode geom = first ranked
            In out chem = other if not identical
            Average: remaining places 
        '''
        if self._Lscore is None:
            self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        chemScore = self._chemScore # Now with same redundancies of large..
        chemScoreS = self._chemScoreS

        avScoreL = 0.5 * (Lscore + chemScore)
        avScoreS = 0.5 * (Sscore + chemScoreS)

        
        indexL = np.where(Lscore<0.5)[0]
        # print('Lscore', Lscore)
        # print(indexL)
        inOutRankGeom = np.argsort(Lscore)[:indexL.size]
        indexC = np.where(chemScore<0.5)[0]
        # print('Cscore', chemScore)
        # print(indexC)
        # print('avScoreL',avScoreL)
        # print(np.argsort(avScoreL))
        inOutRankChem = np.argsort(chemScore)[:indexC.size]
        rank = concatenateRanks(inOutRankGeom,inOutRankChem)
        # print('inout ranks concatenated')
        # print(rank)
        mainrank = concatenateRanks(rank,np.argsort(avScoreL))
        subrank = np.argsort(avScoreS)

        mainScore = np.concatenate([Lscore[:indexL.size],chemScore[:indexC.size]]) #first in/out
        mainScore = np.round(np.concatenate([mainScore,np.sort(avScoreL)[rank.size:]]),3) #remaining. Unique also sorts..
        return mainrank,subrank,np.unique(mainScore),np.sort(avScoreS)


    def specialRanking1_bis(self,featListGeom,featListChem):
        '''
            Average: top3 
            In out mode geom and in/out chem remaining places
            OBSERVATION : could not run up to 10 since I'm using finite subsets of all the candidate pockets
        '''
        if self._Lscore is None:
            self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
        chemScore = self._chemScore # Now with same redundancies of large..
        chemScoreS = self._chemScoreS

        avScoreL = 0.8 * Lscore + 0.2 * chemScore
        avScoreS = 0.8*Sscore + 0.2*chemScoreS


        # LscoreUnique,SscoreUnique = self.getScoresUnique(featListGeom + featListChem)

        
        indexL = np.where(Lscore<0.5)[0]
        # print('Lscore',Lscore)
        # print('Sorted Lscore', np.sort(Lscore)[:10])
        # print(np.argsort(Lscore)[:indexL.size])
        inOutRankGeom = np.argsort(Lscore)[:indexL.size]
        indexC = np.where(chemScore<0.5)[0]
        # print('Cscore',chemScore)
        # print('sorted Cscore', np.sort(chemScore)[:10])
        # print( np.argsort(chemScore)[:indexC.size])
        # print('sorted avScoreL',np.sort(avScoreL)[:10])
        # print(np.argsort(avScoreL)[:10])
        inOutRankChem = np.argsort(chemScore)[:indexC.size]
        
        
        rank = concatenateRanks(np.argsort(avScoreL)[0:3],inOutRankGeom)
        
        # print('inout ranks concatenated')
        # print(rank)
        mainrank = concatenateRanks(rank,inOutRankChem)
        subrank = np.argsort(avScoreS)

        mainScore = np.concatenate([np.sort(avScoreL)[0:3],Lscore[:indexL.size]]) #first in/out
        mainScore = np.round(np.concatenate([mainScore,chemScore[:indexC.size]]),3) #remaining. Unique also sorts..
        return mainrank,subrank,np.unique(mainScore),np.sort(avScoreS)


#2
# instead of average, score-sorted concatenation --> careful need to filter repetitions.. ATTENZIONE: np.unique I don't remember if it also sorts
#ex. Chem = 0.44,0.445,0.47 ; Geom= 0.443,0.45,0.46 -->0.44,0.443.0.445,0.45,0.46,0.47 Allora sufficient to merge two arrays and sort
# and keep track of the original indexed which are used to create a new vector of the 2 ranks which then is filtred for uniqueness (respecting sort)


    def specialRanking2(self,featListGeom,featListChem):
        '''
            Ranking based on concatenation of individual geometrical and chemical forests --> The actual value of the score matters..
            This could be similar to using a unique forest combining chemical and geometrical features..
            
        '''
        if self._Lscore is None:
            self._getScores(featListGeom,featListChem)
        else:
            pass
        
        Lscore = self._Lscore
        npockets = Lscore.size

        Sscore = self._Sscore
        chemScore = self._chemScore # Now with same redundancies of large.. to make sure score comparable..
        chemScoreS = self._chemScoreS # Idem for small 
        concatenateScoreL = np.concatenate([Lscore,chemScore])
        mainScore = np.unique(np.round(concatenateScoreL,3))[:npockets] # already sorted
        concatenateScoreS = np.concatenate([Sscore,chemScoreS])
        subScore = np.unique(np.round(concatenateScoreS,3))[:npockets] # already sorted

        indMain = [i - (npockets) if i>=npockets else i for i in np.argsort(concatenateScoreL)]
        mainRank = orderPreservedUnique(indMain)

        indSub = [i - (npockets) if i>=npockets else i for i in np.argsort(concatenateScoreS)]
        subRank = orderPreservedUnique(indSub)
        # print('Lscore',np.sort(Lscore)[:10])
        # print('Lrank', np.argsort(Lscore))
        # print('ChemScore',np.sort(chemScore)[:10])
        # print('ChemRank', np.argsort(chemScore))

        return mainRank,subRank,mainScore,subScore
        



###################### OLD ###################333
    def getGlobalRanking(self,featListGeom,featListChem):
        if self._Lscore is None:
            self._getScores(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
       
        # Lscore = self._modelL.getScore(featList)
        # Sscore = self._modelS.getScore(featList)
        av_score=0.5*(Lscore+Sscore)
        av_score=np.round(av_score,3)
        rank = np.lexsort((Sscore,av_score)) # --> More sensitive to small p ranking
        self._globalRank = rank
        self._globalScore = av_score
        return rank,rank,np.sort(av_score),np.sort(av_score)

    def getLargeRanking(self,featListGeom,featListChem):
        # if self._Lscore is None:
        #     self._getScores(featList)
        # else:
        #     pass
        
        # Sscore = self._Sscore
        # Lscore = self._modelL.getScore(featList)
        
        if(self._globalRank is None):
            self.getGlobalRanking(featListGeom,featListChem)
        else:
            pass
        Lscore = self._Lscore
        rank = np.argsort(Lscore)
        # Sscore = self._modelS.getScore(featList)
        # av_score=0.5*(Lscore+Sscore)
        # av_score=np.round(av_score,3)
        # globalrank = np.lexsort((Sscore,av_score))
        av_score = self._globalScore
        subrank = self._globalRank
        return rank,subrank,np.sort(np.round(Lscore,3)),np.sort(av_score)

    def getSmallRanking(self,featList):
        # if self._Lscore is None:
        #     self._getScores(featList)
        # else:
        #     pass
        
        # Sscore = self._Sscore
        # Lscore = self._modelL.getScore(featList)
        
        if(self._globalRank is None):
            self.getGlobalRanking(featList)
        else:
            pass
        Sscore = self._Sscore
        rank = np.argsort(Sscore)
        # Sscore = self._modelS.getScore(featList)
        # av_score=0.5*(Lscore+Sscore)
        # av_score=np.round(av_score,3)
        # globalrank = np.lexsort((Sscore,av_score))
        # av_score = self._globalScore
        # subrank = self._globalRank
        return rank,rank,np.sort(np.round(Sscore,3)),np.sort(np.round(Sscore,3)) #subrank coincides..

    def getMixedRanking2(self,featList):
        '''
        First places occupied by large forest IN/OUT. Then small forest score
        '''
        if self._Lscore is None:
            self._getScores(featList)
        else:
            pass
        Lscore = self._Lscore
        Sscore = self._Sscore
       
        index = np.where(Lscore<0.5)[0]
        inOutRank = np.argsort(Lscore)[:index.size]
        small_rank = np.argsort(Sscore)
        final_rank = concatenateRanks(inOutRank,small_rank)
        return final_rank,small_rank,np.unique(np.round(np.concatenate([Lscore[:index.size],Sscore]),3)),np.sort(Sscore)


    def getMixedRanking(self,featList):
        '''
        First places occupied by large forest IN/OUT. Then average between large and small (global score).
        NOTE: This wont be distinguishable with large forest ranking if many scores of the large forest above 0.5
        '''
        if(self._globalRank is None):
            self.getGlobalRanking(featList)
        else:
            pass
        # Lscore = self._modelL.getScore(featList)
        # Sscore = self._modelS.getScore(featList)
        # av_score=0.5*(Lscore+Sscore)
        # av_score=np.round(av_score,3)
        # global_rank = np.lexsort((Sscore,av_score))
        Lscore = self._Lscore
        global_rank = self._globalRank
        av_score = self._globalScore
        index = np.where(Lscore<0.5)[0]
        # print(np.argsort(Lscore)[:10])
        # print(np.sort(Lscore)[:10])
        inOutRank = np.argsort(Lscore)[:index.size]
        # print(inOutRank)
        final_rank = concatenateRanks(inOutRank,global_rank)
        subrank = global_rank
        return final_rank,subrank,np.unique(np.round(np.concatenate([Lscore[:index.size],av_score]),3)),np.sort(av_score)
###########################################################################
####################################

def getRank(pList,resMap,name, mode,loadTrained = True,structureName='',isPedro=False,noSorting=False):
    """
    USer oriented ranking function
    """
    err = Error()
    
    # Note a ranked subpocket does NOT promote the parent pocket; a parent pocket, includes the subpocket which is skipped in the main ranking.
    # An ecception if for single subpockets, which provokes skipping of master pocket in the ranking
    # For user purposes this makes sense.

    # 1. Loading trained IFs or train IFs using a feature container
    if loadTrained:
        score = Scoring()
        if(re.match(".*IF\d*",name)):
            m = ML["IsolationForest"]
        elif(re.match(".*EIF\d*",name)):
             m = ML["Extended IsolationForest"]
        else: 
            print("Model not implemented..")
            print(name)
            err.value=2
            err.info = "Model not implemented"
            
            exit()
    else:
        print("Embedded training not implemented..")
        err.value=2
        err.info = "Embedded training not implemented"
        
        exit()
    if(mode==8):
        err = score.load(name+'_unique',modelType=m,unique=True,useCNN=isPedro)
    else:
        err = score.load(name,modelType=m,useCNN=isPedro)

    if(err.value==2):
        print("<<ERROR>> Cannot load trained model\n Aborting.")
        exit()

    if(isPedro):
        print("Using also CNN features")
    else:
        print("Using residues and hydro scores for chemical properties")
    #**************************************
    
    # featListGeom = [] #Entry number must be the pocket number at the end with the number = the ranking
    # featListChem = []

    # mapd={} #map general index-->pocket and subpocket indexes
    # mapr={} #map pocket and subpocket indexes --> general index
    # s = 0
    # for parentPindex,p in enumerate(pList):
    #     # print(parentPindex)
    #     err,fg,fc = buildTestFeatures(p,resMap)
    #     if(err.value==1):
    #         #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
    #         print("Skipping pocket, since unable to estimate the volume")
    #         continue
    #     featListGeom.append(fg)
    #     featListChem.append(fc)

    #     mapd[s] = (parentPindex,None)
    #     mapr [(parentPindex,None)]=s
    #     s+=1
    #     for subIndex,sub in enumerate(p['subpockets']):
    #         err,fg,fc = buildTestFeatures(sub,resMap,imSub=True)
    #         if(err.value==1):
    #             print("Skipping sub-pocket, since unable to estimate the volume")
    #             continue
    #         featListGeom.append(fg)
    #         featListChem.append(fc)
    #         mapd[s] = (parentPindex,subIndex)
    #         mapr [(parentPindex,subIndex)]=s
    #         s+=1
    # featList = [fg + fc for fg,fc in zip(featListGeom,featListChem)]
    if(isPedro == False):
        featListGeom,featListChem,map_direct,map_reverse = getFeatNoDensity(pList,resMap)
    else:
        featListGeom,featListChem,featListPedro,map_direct,map_reverse = getFeatPedro(pList,resMap,structureName=structureName)
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRankingPedro(featListGeom,featListChem,featListPedro)
        return_featList = featListGeom
        return (rankedIndexes,numericalScore),(rankedIndexesSub,numericalScoreSub),return_featList,map_direct,map_reverse
    return_featList = featListGeom
    if(mode ==1): 
        print("\n**50/50 geometry and chemistry ranking mode**\n")
        if(not noSorting):
            rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRanking(featListGeom,featListChem)
        else:
            rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRanking_noSorting(featListGeom,featListChem)
        # if(name=="IF10_noDensity"):
        #     rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRanking(featListGeom,[f[:22] for f in featListChem])
        # else:
        #     rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRanking(featListGeom,featListChem)
    if(mode ==2): 
        print("\n**80/20 geometry and chemistry ranking mode**\n")
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRanking8020(featListGeom,featListChem)
    if(mode ==3): 
        print("\n**Only geometry ranking mode**\n")
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRankingOnlyGeom(featListGeom)
    if(mode ==4): 
        print("\n**Only chemistry ranking mode**\n")
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRankingOnlyChem(featListChem)

    if(mode ==5): 
        print("\n**SPECIAL RANKING 1**\n")
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.specialRanking1(featListGeom,featListChem)

    if(mode ==6): 
        print("\n**SPECIAL RANKING 1 BIS**\n")
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.specialRanking1_bis(featListGeom,featListChem)
    if(mode ==7): 
        print("\n**SPECIAL RANKING 2**\n")
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.specialRanking2(featListGeom,featListChem)

    if(mode==8):
        print('UNIQUE CHEM GEOM FOREST')
        rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getRankingUnique([fg + fc for fg,fc in zip(featListGeom,featListChem)])
    # if(mode ==1): #GLOBAL RANKING MODE
    #     print("\n**GLOBAL RANKING MODE**\n")
    #     rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getGlobalRanking(featList)
    # if(mode==2): #LARGE P RANKING MODE
    #     print("\n**LARGE P RANKING MODE**\n")
    #     rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getLargeRanking(featList)
    # if(mode==3): #SPECIAL RANKING MIXING BOTH MODES: First ranked with L-model in IN/OUT mode, then remaining places with global 
    #             # CAREFUL: must filter for uniqueness: check that first appearence is what keeps the ranking
    #     print("\n**MIXED RANKING MODE**\n")
    #     rankedIndexes,rankedIndexesSub,numericalScore,numericalScoreSub = score.getMixedRanking(featList)
    #Now I have a array of ranked indexes like: [0,3,4,2,7,13]

    return (rankedIndexes,numericalScore),(rankedIndexesSub,numericalScoreSub),return_featList,map_direct,map_reverse

def getFeat(pList,resMap):
    '''
    Output: List of list. First index features per pocket, second index spans the features
    '''
    featListGeom = [] 
    featListChem = []
    mapd={} 
    mapr={}
    s = 0
    for parentPindex,p in enumerate(pList):
        # print(parentPindex)
        err,fg,fc = buildTestFeatures(p,resMap)
        if(err.value==1):
            #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
            print("Skipping pocket, since unable to estimate the volume")
            continue
        featListGeom.append(fg)
        featListChem.append(fc)
        mapd[s] = (parentPindex,None)
        mapr [(parentPindex,None)]=s
        s+=1
        for subIndex,sub in enumerate(p['subpockets']):
            err,fg,fc = buildTestFeatures(sub,resMap,imSub=True)
            if(err.value==1):
                print("Skipping sub-pocket, since unable to estimate the volume")
                continue
            featListGeom.append(fg)
            featListChem.append(fc)
            mapd[s] = (parentPindex,subIndex)
            mapr [(parentPindex,subIndex)]=s
            s+=1
        
    return featListGeom,featListChem,mapd,mapr

def getFeatNoDensity(pList,resMap):
    '''
    Output: List of list. First index features per pocket, second index spans the features
    '''
    featListGeom = [] 
    featListChem = []
    mapd={} 
    mapr={}
    s = 0
    for parentPindex,p in enumerate(pList):
        # print(parentPindex)
        err,fg= buildGeomFeatures(p)
        err,fc= buildChemFeaturesNoDensity(p,resMap)
        if(err.value==1):
            #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
            print("Skipping pocket, since unable to estimate the volume")
            continue
        featListGeom.append(fg)
        featListChem.append(fc)
        mapd[s] = (parentPindex,None)
        mapr [(parentPindex,None)]=s
        s+=1
        for subIndex,sub in enumerate(p['subpockets']):
            err,fg = buildGeomFeatures(sub,imSub=True)
            err,fc= buildChemFeaturesNoDensity(sub,resMap)
            if(err.value==1):
                print("Skipping sub-pocket, since unable to estimate the volume")
                continue
            featListGeom.append(fg)
            featListChem.append(fc)
            mapd[s] = (parentPindex,subIndex)
            mapr [(parentPindex,subIndex)]=s
            s+=1
        
    return featListGeom,featListChem,mapd,mapr

def getFeatPedro(pList,resMap,structureName):
    '''
    Output: List of list. First index features per pocket, second index spans the features
    '''
    

    featListGeom = [] 
    featListChem = []
    featListPedro =[]
    mapd={} 
    mapr={}
    s = 0

    pedroFeat = CNNFeature()
    err = pedroFeat.init(structureName)
    # print('HERE')
    if err.value==2:
        raise Exception("<<ERROR>>"+err.info)
    for parentPindex,p in enumerate(pList):
        # print(parentPindex)
        err,fg= buildGeomFeatures(p)
        if(err.value==1):
            #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
            print("Skipping pocket, since unable to estimate the volume")
            continue
        err,fc= buildChemFeaturesNoDensity(p,resMap) #Dummy error, never produced
        err,fp= pedroFeat.build(p,resMap)
        if(err.value==1):
            #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
            print("Skipping pocket, since NaN in CNN features")
            continue
        
        featListGeom.append(fg)
        featListPedro.append(fp)
        featListChem.append(fc)
        mapd[s] = (parentPindex,None)
        mapr [(parentPindex,None)]=s
        s+=1
        for subIndex,sub in enumerate(p['subpockets']):
            err,fg = buildGeomFeatures(sub,imSub=True)
            if(err.value==1):
                print("Skipping sub-pocket, since unable to estimate the volume")
                continue
            err,fc= buildChemFeaturesNoDensity(sub,resMap)
            err,fp= pedroFeat.build(sub,resMap)
            if(err.value==1):
                #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
                print("Skipping pocket, since NaN in CNN features")
                continue
            
            featListGeom.append(fg)
            featListPedro.append(fp)
            featListChem.append(fc)
            mapd[s] = (parentPindex,subIndex)
            mapr [(parentPindex,subIndex)]=s
            s+=1
        
    return featListGeom,featListChem,featListPedro,mapd,mapr

def getFeatOnlyDensity(pList,resMap):
    '''
    Output: List of list. First index features per pocket, second index spans the features
    '''
    featListGeom = [] 
    featListChem = []
    mapd={} 
    mapr={}
    s = 0
    for parentPindex,p in enumerate(pList):
        # print(parentPindex)
        err,fg= buildGeomFeatures(p)
        err,fc= buildChemFeaturesDensity(p,resMap)
        if(err.value==1):
            #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
            print("Skipping pocket, since unable to estimate the volume")
            continue
        featListGeom.append(fg)
        featListChem.append(fc)
        mapd[s] = (parentPindex,None)
        mapr [(parentPindex,None)]=s
        s+=1
        for subIndex,sub in enumerate(p['subpockets']):
            err,fg = buildGeomFeatures(sub,imSub=True)
            err,fc= buildChemFeaturesDensity(sub,resMap)
            if(err.value==1):
                print("Skipping sub-pocket, since unable to estimate the volume")
                continue
            featListGeom.append(fg)
            featListChem.append(fc)
            mapd[s] = (parentPindex,subIndex)
            mapr [(parentPindex,subIndex)]=s
            s+=1
        
    return featListGeom,featListChem,mapd,mapr

def getFeatAlt(pList,resMap):
    '''
    Output: List of list. First index features per pocket, second index spans the features
    '''
    featListGeom = [] 
    featListChemDensity = []
    featListChemNoDensity =[]
    mapd={} 
    mapr={}
    s = 0
    for parentPindex,p in enumerate(pList):
        # print(parentPindex)
        err,fg= buildGeomFeatures(p)
        err,fcDensity= buildChemFeaturesDensity(p,resMap)
        err,fcNoDensity= buildChemFeaturesNoDensity(p,resMap)
        if(err.value==1):
            #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
            print("Skipping pocket, since unable to estimate the volume")
            continue
        featListGeom.append(fg)
        featListChemDensity.append(fcDensity)
        featListChemNoDensity.append(fcNoDensity)
        mapd[s] = (parentPindex,None)
        mapr [(parentPindex,None)]=s
        s+=1
        for subIndex,sub in enumerate(p['subpockets']):
            err,fg = buildGeomFeatures(sub,imSub=True)
            err,fcDensity= buildChemFeaturesDensity(sub,resMap)
            err,fcNoDensity= buildChemFeaturesNoDensity(sub,resMap)
            if(err.value==1):
                print("Skipping sub-pocket, since unable to estimate the volume")
                continue
            featListGeom.append(fg)
            featListChemDensity.append(fcDensity)
            featListChemNoDensity.append(fcNoDensity)
            mapd[s] = (parentPindex,subIndex)
            mapr [(parentPindex,subIndex)]=s
            s+=1
        
    return featListGeom,featListChemDensity,featListChemNoDensity,mapd,mapr


def getFeatUnique(pList,resMap):
    '''
    Output: List of list. First index features per pocket, second index spans the features
    '''
    featList=[]
    mapd={} 
    mapr={}
    s = 0
    for parentPindex,p in enumerate(pList):
        # print(parentPindex)
        err,f = buildTestFeaturesUnique(p,resMap)
        if(err.value==1):
            #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
            print("Skipping pocket, since unable to estimate the volume")
            continue
        featList.append(f)
    
        mapd[s] = (parentPindex,None)
        mapr [(parentPindex,None)]=s
        s+=1
        for subIndex,sub in enumerate(p['subpockets']):
            err,f = buildTestFeaturesUnique(sub,resMap,imSub=True)
            if(err.value==1):
                print("Skipping sub-pocket, since unable to estimate the volume")
                continue
            featList.append(f)
            
            mapd[s] = (parentPindex,subIndex)
            mapr [(parentPindex,subIndex)]=s
            s+=1
        
    return featList,mapd,mapr

def getRank_loop(featList,model1):
    #OBS: truncation of single subPockets is not necessary since a filtering for redundant classification can be performed outside in case
    # Still, might be interesting to establish if a subpocket of a pockets hits.
    # Indeed a possible scenario is that eg. p0 has sub1,2,3,4,5. p0 is high in ranking but does not hit. But sub2 its. Maybe, a simple rule
    # which establish that the pocket i considered succesfull if one of its 3 topmost ranked subpockets hit.
    # Keep this as a separate evaluator, to keep information if the master pocket is already good
    # Therefore I need to have a way to reconstruct relative ranking of subpockets
    
    """
    The model is loaded externally
    Only returns ranking and reconstrution of ranked pockets (not the extracted features).
    Performs at once many types of ranking for different ML models
    Buried, activates the option to add the extra buried feature
    """
    # Rank according to different ranking modes

    # featList = [] 
    # mapd={} 
    # mapr={}
    # s = 0
    # for parentPindex,p in enumerate(pList):
    #     print(parentPindex)
    #     err,f = buildTestFeatures(p)
    #     if(err.value==1):
    #         #Maybe in these scenarios assume 0 volume? (but is the same, it will cause the pocket to be out from ranking)
    #         print("Skipping pocket, since unable to estimate the volume")
    #         continue
    #     featList.append(f)
    #     mapd[s] = (parentPindex,None)
    #     mapr [(parentPindex,None)]=s
    #     s+=1
    #     for subIndex,sub in enumerate(p['subpockets']):
    #         err,f = buildTestFeatures(sub)
    #         if(err.value==1):
    #             print("Skipping sub-pocket, since unable to estimate the volume")
    #             continue
    #         featList.append(f)
    #         mapd[s] = (parentPindex,subIndex)
    #         mapr [(parentPindex,subIndex)]=s
    #         s+=1
    # print("Built Features")
    # featList,mapd,mapr = getFeat(pList,buried) DONE EXTERNALLY
    model1.resetRank()
    # model2.resetRank()
    returnDict = {}
    #GLOBAL RANKING MODE OVER
    #NOW PURE SMALL RANKING
    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getGlobalRanking(featList)
    # rankedIndexesEIF,rankedIndexesSubEIF,numericalScoreEIF,numericalScoreSubEIF = model2.getGlobalRanking(featList)
    # returnDict[1]=[(rankedIndexesIF,rankedIndexesSubIF),(rankedIndexesEIF,rankedIndexesSubEIF)]
    returnDict[1]=(rankedIndexesIF,rankedIndexesSubIF)

    #LARGE P RANKING MODE
    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getLargeRanking(featList)
    # rankedIndexesEIF,rankedIndexesSubEIF,numericalScoreEIF,numericalScoreSubEIF = model2.getLargeRanking(featList)
    # returnDict[2]=[(rankedIndexesIF,rankedIndexesSubIF),(rankedIndexesEIF,rankedIndexesSubEIF)]
    returnDict[2]=(rankedIndexesIF,rankedIndexesSubIF)

    #RANKING MIXING BOTH MODES: First ranked with L-model in IN/OUT mode, then remaining places with global 
    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getMixedRanking(featList)
    # rankedIndexesEIF,rankedIndexesSubEIF,numericalScoreEIF,numericalScoreSubEIF = model2.getMixedRanking(featList)
    # returnDict[3]=[(rankedIndexesIF,rankedIndexesSubIF),(rankedIndexesEIF,rankedIndexesSubEIF)]
    returnDict[3]=(rankedIndexesIF,rankedIndexesSubIF)

    #RANKING MIXING BOTH MODES: First ranked with L-model in IN/OUT mode, then remaining places with SMALL 
    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getMixedRanking2(featList)
    returnDict[4]=(rankedIndexesIF,rankedIndexesSubIF)

    #CAREFUL TODO: outside I must still check for redundancies (if master pocket higher in ranking) 
    return returnDict


def getRank_loopWithChem(featListGeom,featListChem,model1):

    model1.resetRank()

    returnDict = {}

    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getRanking(featListGeom,featListChem)
    returnDict[1]=(rankedIndexesIF,rankedIndexesSubIF)


    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getRankingOnlyGeom(featListGeom)
    returnDict[2]=(rankedIndexesIF,rankedIndexesSubIF)

   
    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getRankingOnlyChem(featListChem)
    returnDict[3]=(rankedIndexesIF,rankedIndexesSubIF)


    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getRanking8020(featListGeom,featListChem)
    returnDict[4]=(rankedIndexesIF,rankedIndexesSubIF)

    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.specialRanking1(featListGeom,featListChem)
    returnDict[5]=(rankedIndexesIF,rankedIndexesSubIF)

    return returnDict

def getRank_loopSpecial(featListGeom,featListChem,model1):

    model1.resetRank()

    returnDict = {}

    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.specialRanking1(featListGeom,featListChem)
    returnDict[1]=(rankedIndexesIF,rankedIndexesSubIF)

    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.specialRanking1_bis(featListGeom,featListChem)
    returnDict[2]=(rankedIndexesIF,rankedIndexesSubIF)

    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.specialRanking2(featListGeom,featListChem)
    returnDict[3]=(rankedIndexesIF,rankedIndexesSubIF)

    return returnDict

def getRank_loopUnique(featList,model1):

    model1.resetRank()

    returnDict = {}

    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getRankingUnique(featList)
    returnDict[1]=(rankedIndexesIF,rankedIndexesSubIF)

    rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = model1.getRankingUniqueAverage(featList)
    returnDict[2]=(rankedIndexesIF,rankedIndexesSubIF)


    return returnDict


def rerankVol_withSub(originalPlist,rankingForest,scoresIF,map_direct,map_reverse,keep=10,cutoff=1):
    '''
    New ranking method which uses forest for top10 and volume among them. But among top10 subpockets are not counted as filling the ranking.
    (r index not progressing)
    '''
    r=0
    r_in=0
    selfContained = set()
    alreadyDoneSub=set()
    pListTop = []
    vols =[]
    scores=[]

    num_sub=0
    
    while ((r<10) and (r_in < rankingForest.size)and (scoresIF[rankingForest[r_in]]<=cutoff)):
        pi,si = map_direct[rankingForest[r_in]]
        
        if(si is not None):
    # ------------------- SUBPOCKET--------------------------- 
            if(pi in selfContained):
                #skip subpocket of a master pocket ahead in the ranking
                #going to next element (positions do not progress in the r
                r_in+=1
                continue # r index does not advance
            n_subs=0
            pocket = originalPlist[pi]['subpockets'][si] #IS A SUBPOCKET
            pListTop.append(pocket)
            volume,_A,_err = pocket['node'].volume()
            vols.append(volume)
            scores.append(scoresIF[map_reverse[(pi,si)]])

            if(len(originalPlist[pi]['subpockets'])==1):
                #Single subpocket--> master pocket in black list
                selfContained.add(pi)
            else:
                alreadyDoneSub.add((pi,si))
        else:
            # --------- PARENT POCKET (or with no subpockets) --------------------
            if(pi in selfContained): #SKIP PARENT POCKET OF A SINGLE SUBPOCKET ALREADY EXPRESSED IN THE RANKING
                r_in+=1
                continue
            selfContained.add(pi) #to filter out subpockets already expressed by the master pocket
            
            subs = originalPlist[pi]['subpockets']
            n_subs = len(subs)
            # print('Master pocket with %d subpockets'%n_subs)
            # pocket = originalPlist[pi]
            # vol,_A,_err = pocket['node'].volume()
            # print('Master volume = ',vol)
            if(n_subs==0):
                #pocket with no subpockets
                pocket = originalPlist[pi]
                pListTop.append(pocket)
                volume,_A,_err = pocket['node'].volume()
                vols.append(volume)
                scores.append(scoresIF[map_reverse[(pi,si)]])
            for i in range(n_subs):
                if((pi,i)in alreadyDoneSub):
                    # I think this scenario is very unlikely
                    pass
                else:
                    num_sub+=1
                    subpocket = subs[i]
                    pListTop.append(subpocket)
                    volume,_A,_err = subpocket['node'].volume()
                    vols.append(volume)
                    scores.append(scoresIF[map_reverse[(pi,i)]])
                    # print('sub%d, vol=%.2f'%(i,volume))

        r+=1
        r_in+=1
        
    print('Original number of pockets considered',len(vols))
    print('number of subpockets opened:',num_sub)
    # print(vols)
    ind = np.argsort(vols)[::-1]
    pListTop = np.array(pListTop)[ind]
    vols = np.array(vols)[ind]
    print('sorted volumes:',vols)

    return pListTop[:keep],scores


def rerankVol_noSub(originalPlist,rankingForest,scoresIF,map_direct,map_reverse,keep=10,cutoff=1):
    '''
    New ranking method which uses forest for top10 and volume among them. But among top10 subpockets are not counted as filling the ranking.
    (r index not progressing)
    NOTE: Not skipping master pocket of single subpocket already expressed. This because in any case the volume will bring forward the master pocket.
    '''
    r=0
    r_in=0
    selfContained = set()
    alreadyDoneSub=set()
    pListTop = []
    vols =[]
    scores=[]
    while ((r<10) and (r_in < rankingForest.size) and (scoresIF[rankingForest[r_in]]<=cutoff)):
        pi,si = map_direct[rankingForest[r_in]]
        if(si is not None):
    # ------------------- SUBPOCKET--------------------------- 
            if(pi in selfContained):
                #skip subpocket of a master pocket ahead in the ranking
                #going to next element (positions do not progress in the r
                r_in+=1
                continue # r index does not advance
            
            pocket = originalPlist[pi]['subpockets'][si] #IS A SUBPOCKET
            pListTop.append(pocket)
            volume,_A,_err = pocket['node'].volume()
            vols.append(volume)
            
            scores.append(scoresIF[map_reverse[(pi,si)]])
            # if(len(originalPlist[pi]['subpockets'])==1):
            #     #Single subpocket--> master pocket in black list
            #     selfContained.add(pi)
        else:
            # --------- PARENT POCKET (or with no subpockets) --------------------
            # if(pi in selfContained): #SKIP PARENT POCKET OF A SINGLE SUBPOCKET ALREADY EXPRESSED IN THE RANKING
            #     r_in+=1
            #     continue
            selfContained.add(pi) #to filter out subpockets already expressed by the master pocket
            pocket = originalPlist[pi]
            pListTop.append(pocket)
            volume,_A,_err = pocket['node'].volume()
            vols.append(volume)
            scores.append(scoresIF[map_reverse[(pi,si)]])
            

        r+=1
        r_in+=1
        
    
    # print(vols)
    print('Original number of pockets considered keeping only master pockets or single clusters',len(vols))
    ind = np.argsort(vols)[::-1]
    vols = np.array(vols)[ind]
    pListTop = np.array(pListTop)[ind]
    return pListTop[:keep],scores






#2
# instead of average, score-sorted concatenation --> careful need to filter repetitions.. ATTENZIONE: np.unique I don't remember if it also sorts
#ex. Chem = 0.44,0.445,0.47 ; Geom= 0.443,0.45,0.46 -->0.44,0.443.0.445,0.45,0.46,0.47 Allora sufficient to merge two arrays and sort
# and keep track of the original indexed which are used to create a new vector of the 2 ranks which then is filtred for uniqueness (respecting sort)