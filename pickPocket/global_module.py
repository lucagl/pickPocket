
# Time profiling routines and modules 
import numpy as np
from time import time, strftime, localtime
from datetime import timedelta

import os 
import math
import re 
from sys import exit

import pickPocket

##################


################################
#############


#CLUSTERING PARAMETERS
shift_Threshold = 4
count_Threshold = 100 
weight = count_Threshold
rmin = 0 #To enforce minimum radius exit
rmin_entrance = 2.5

lightMode=False

delta_rp = 0.1  #THIS IS A PARAMETER WITHIN THE CLUSTERING ALGORITHM

largeP=2
tollerance = 5.e-2 

accTriang = True

##########
PI = np.pi
R_WATER = 1.4
WATER_R3 = R_WATER*R_WATER*R_WATER
MIN_VOL = (4./3.*PI*R_WATER)
sin15_2 =   math.sin(math.pi/24.) 



libname = os.path.dirname(pickPocket.__file__)+'/libCfunc.so'
if os.path.exists(libname):
    # print(libname)
    pass
else:
    exit("\'libCfunc.so\' must be present in "+os.path.dirname(pickPocket.__file__))

trainingData = os.path.abspath(os.path.dirname(pickPocket.__file__)+'/trainedModels')+'/'  
if os.path.isdir(trainingData):
    # print(trainingData)
    pass
else:
    exit("trainedModels folder  must be present in "+os.path.dirname(pickPocket.__file__))

pathTo_NS_ex=os.path.abspath(os.path.dirname(pickPocket.__file__)+'/refTemp')+'/'
if os.path.isdir(pathTo_NS_ex):
    # print(pathTo_NS_ex)
    pass
else:
    exit("NS executable and input must be present in"+os.path.dirname(pickPocket.__file__))
#TODO: run a test of NS to check if libraries correctly linked..

pdbFolder_path = os.path.abspath(".")+'/'
runFolder_path = os.path.abspath(".")+'/'
save_path = pdbFolder_path
conf = 'surfaceConfiguration.prm' #Get from input file in the future

### ACCESSORY FUNCTIONS ##
def initFolders(pdbName=None):
    
    global pdbFolder_path
    global runFolder_path
    global save_path
    default_pdbFolderName = "structures"
    default_runFolderName = "temp"
    err = Error()
    here = os.path.abspath(".")+'/'
    isFolder = os.path.isdir(here+default_runFolderName)
    if(not isFolder):
        try:
            raise FileNotFoundError
        except FileNotFoundError as info:
            print("\n ** Did not find folder where to store temp files \'temp\' ")
            err.info = str(type(info)) + "Cannot find folder where to store temp files"
            err.value = 2
            return err
    else:
        runFolder_path = os.path.abspath(default_runFolderName)+'/'

    ###CHECK MAIN NS FILES ARE PRESENT 
    isFile = os.path.isfile(runFolder_path+conf)
    if(not isFile):
        try:
            raise FileNotFoundError
        except FileNotFoundError as info:
            print("\n**Did not find NanoShaper configuration file: 'surfaceConfigurattion.prm' ")
            err.info = str(type(info)) + "Did not find NanoShaper configuration file:'surfaceConfigurattion.prm' "
            err.value = 2
            return err
    isFile = os.path.isfile(runFolder_path+"NanoShaper")
    if(not isFile):
        try:
            raise FileNotFoundError
        except FileNotFoundError as info:
            print("\n**Did not find NanoShaper executable ")
            err.info = str(type(info)) + "Did not find NanoShaper executable "
            err.value = 2
            return err
    

    ##


    isFolder = os.path.isdir(here+default_pdbFolderName)
    if(not isFolder):
        try:
            raise FileNotFoundError
        except FileNotFoundError as info:
            # print("Did not find folder containing structures to analyse. Setting working directory.")
            err.info = str(type(info)) + "Cannot find folder containing structure. Setting working directory."
            err.value = 1
        pdbFolder_path = os.path.abspath(".")+'/'
    else:
        pdbFolder_path = os.path.abspath(default_pdbFolderName)+'/'
    # print(runFolder_path)
    if(pdbName!=None):
        isFile = os.path.isfile(pdbFolder_path+pdbName+".pqr")
        if(not isFile):
            try:
                raise FileNotFoundError
            except FileNotFoundError as info:
                print("\n** Did not find pqr file. CAREFUL: if structure folder is provided, the file must be there.")
                err.info = str(type(info)) + "Did not find pqr file  "
                err.value = 2
                return err
    save_path = pdbFolder_path
    return err


def secondsToStr(elapsed=0):
    if elapsed ==0 :
        return strftime("LOCAL TIME = %Y-%m-%d %H:%M:%S", localtime())
    else:
        return "ELAPSED TIME: "+ str(timedelta(seconds=elapsed))


class Crono(object):
    def init(self):
        self._elapsed=0
        self._start=time()
        return secondsToStr(self._elapsed)
    def get(self):
        end=time()
        self._elapsed=end-self._start
        return secondsToStr(self._elapsed)


################### USEFULL CLASSES AND METHODS #####################

comment = ['^#','\n']
comment = '(?:% s)' % '|'.join(comment)
_warning = "<WARNING> "
_break = "<ERROR> "
class Error(object):
    n_errors = 0
    def __init__(self):
        self.value = 0
        self.info = " "
        self._status = None
    def reset(self):
        self.__init__()
    def build_status(self):
        if(self.value==1):
            self._status = _warning
        elif (self.value==2):
            self._status = _break
        else:
            pass
        return
    def get_info(self):
        self.build_status()
        # print(self._status + self.info)
        return self._status + self.info
    def write(self,errFile):
        try:
            errFile.write("\n "+self.get_info())
        except OSError:
            raise NotImplementedError("Error log file not found..")
        return
    
    def handle(self,errFile):
        #OBS: Only when errors are handled the static counter is updated
        if(self.value==0): #do nothing
            self.reset()
            return  
        Error.n_errors +=1 
        try:
            errFile.write("\n "+self.get_info())
        except OSError:
            raise NotImplementedError("Error log file not found..")
        if(self.value == 2):
            errFile.close()
            try:
                raise Exception("Exiting for major error")
            except Exception:   
                print("Exiting for major error")
                exit()
        self.reset()#reinitialise to 0 none error message
        # print("CHECK: errror=",self.value,self.info)
        return

############## CLASSS FOR READING FILE 

class ReadConfig(object):
    #read onfiguration file 
    def __init__(self,alpha=0.0,beta=0.6,rpMAX=3):
        self.alpha = alpha
        self.beta = beta
        self.rp_max = rpMAX
        
    def get(self,fileobj):
        err = Error()
        content = fileobj.readlines()
        gotMatch = np.ones(3,bool)*False
        for line in content:
            match1 = re.match("(alpha\s*=\s*)(\d*\.?\d+)",line)
            match2 = re.match("(beta\s*=\s*)(\d*\.?\d+)",line)
            match3 = re.match("(rpMAX\s*=\s*)(\d*\.?\d+)",line)
            if match1:
                gotMatch[0]=True
                try:
                    self.alpha =float(match1.group(2))
                except ValueError as info:
                    err.info = str(type(info))+str(info.args) 
                    err.value =2 
                    return err
            if match2:
                gotMatch[1]=True
                try:
                    self.beta =float(match2.group(2))
                except ValueError as info:
                    err.info = str(type(info))+str(info.args) 
                    err.value =2 
                    return err
            if match3:
                gotMatch[2]=True
                try:
                    self.rp_max =float(match3.group(2))
                except ValueError as info:
                    err.info = str(type(info))+str(info.args) 
                    err.value =2 
                    return err
            if(not gotMatch.any()):
                err.value=2
                err.info="config file is empty. Remove it if you want to use default parameters"
            elif (not gotMatch.all()):
                err.value=1
                err.info="config file partially filled. Some of the parameters set to default"
        return err

###
class ReadInput(object):
    def __init__(self):
        self.alphamax=None
        self.alphamin=None
        self.betamax=None
        self.betamin=None
        self.rpMAX = None
        self.rpMIN = None
        self.delta_alpha = 0.1
        self.delta_beta = 0.1
        self.r_increment = 0.1
        self.mapFile = None
        self.singlePDBname = None 

    def getStructureIterator(self,single=False):
        
        """
        Returns an iterator over structures and ligands
        If single = True, returns simply a single pdb filename.

        Format of the mapLigand file:
        ith line: #ligands    structure pqr name (without .pqr extension specified)
        ith+1-->ith+ligand number   1 line per ligand name UPPER case (convention to distinguish from the structure pqr)

        """
        
        if(single):
            return self.singlePDBname
        
        structures=[]
        content = self.mapFile.readlines()
        
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
                # isPeptide = content[s+i].split()[1]
                if(re.match(comment,following_line)):
                    #skipping ligand
                    continue
                # ligand_names.append((following_line,isPeptide))
                ligand_names.append(following_line)
            structures.append({'pqr':name,'ligands': ligand_names})
            s+=n_ligands +1
            
                
        return structures

    def get(self,fileobj):
        
        from pickPocket.functions import crange

        fail = (None,None,None)
        err = Error()
        content = fileobj.readlines()
        print("\nReading input file\n")
        Analysis = False
        Test = False
        gotMatch=False
        deploy=False
        for line in content:
            # if(re.match(comment,line)): 
            #     continue 
            match = re.match("(Action\s*=\s*)([\S]*)",line)
            # match2 = re.match("(light ?_?mode\s*=\s*)([\S]*)",line)
            # print (line)
            if match:
                gotMatch=True
                if ((match.group(2) == "analyse")or (match.group(2) == "Analyse") or (match.group(2) == "analysis") or (match.group(2) == "Analysis")):
                    print("**Analysis mode\n")
                    Analysis = True

                elif((match.group(2) == "test")or(match.group(2) == "Test")or(match.group(2) == "test_ranking")):
                    print("**Testing performance of ranking\n")
                    Test = True
                else:
                    # print('here')
                    deploy = True
            # if match2:
            #     # print(match2.group(2))
            #     if ((match2.group(2) == "True")or (match2.group(2) == "1")or (match2.group(2) == "true")or (match2.group(2) == "yes")):
            #         global_module.lightMode = True
        if not gotMatch:
            try:
                raise EOFError("The field \'Action\' is not present in the input file")
            except EOFError as info:
                err.value=2
                err.info=str(type(info)) + "The field \'Action\' is not present in the input file"
                print("The field \'Action\' is not present in the input file")
                return err,-1,-1,fail 
        # print(deploy)

        if(deploy):
            
            global largeP
            global accTriang
            gotMatch = False
            for line in content:
                # print(line)
                # if(re.match(comment,line)): 
                #     continue 
                match = re.match("(pqr file\s*=\s*)([\S]*)",line)
                match2 = re.match("(largeP filter\s*=\s*)(\d*\.?\d+)",line)
                match3 = re.match("(build structure triang\s*[=:]*\s*)([\w]*)",line)
                if match:
                    gotMatch = True
                    print("PQR file=",match.group(2))
                    self.singlePDBname=match.group(2)
                if match2:
                    #facultative
                    largeP = float(match2.group(2))
                    print("Large pocket overwritten: number of elements=",int(largeP * count_Threshold))
                if match3:
                    #facultative
                    # print("HERE")
                    if ((match3.group(2) == "True") or (match3.group(2)=="true")):
                        accTriang =True
            if not gotMatch:
                    try:
                        raise EOFError("The field pqr file is not present in the input file.\n NOTE: the pqr file can be passed to inline directly.")
                    except EOFError as info:
                        err.value=2
                        err.info=str(type(info)) + "The field pqr file is not present in the input file"
                        print("The field pqr file is not present in the input file")
                        return err,-1,-1,fail
                    
            #Skipping rest of input file and looking for config file
            err = initFolders(self.singlePDBname)
            if(err.value==2):
                return err,-1,fail
            config = ReadConfig()
            try:
                confFile=open("config.txt",'r')
                err=config.get(confFile)
                if(err.value ==2):
                    return err,-1,-1,fail

            except FileNotFoundError:
                print("Config file not found, using defaut parameters")
                err.info = "Config file not found, using defaut parameters"
                err.value = 1
            if(config.rp_max<rmin):
                        err.value = 1
                        err.info = err.info + " Max probe radius cannot be smaller than the minimal radius for a pocket exit, "+str(rmin)+".\n Setting rpMAX =" +str(rmin)
                        config.rp_max = rmin
            return err,False,False,(config.alpha,config.beta,config.rp_max)

#       +++++++++++++ One of the analysis modes +++++++++++++
        else: 
            err = initFolders()
            if(err.value==1):
                err.value=2
                err.info="For this type of running mode a \'structures\' folder containing pqr files is compulsory. Aborting"
                return err,-1,fail
            if(err.value ==2):
                return err,-1,-1,fail
            #Check existence of mapfile
            isFile = os.path.isfile(pdbFolder_path+"ligandMap.txt")
            if(not isFile):
                try:
                    raise FileNotFoundError("ligandMap.txt file not found. This file must be produced using dedicated script..")
                except FileNotFoundError as info:
                    err.info = str(type(info))+str(info.args) 
                    err.value = 2 
                    print("\'ligandMap.txt\' file not found. This file must be produced using dedicated script..")
                    return err,-1,-1,fail
            else:
                self.mapFile = open(pdbFolder_path+"ligandMap.txt",'r')
            gotMatch = np.ones(6,bool)*False
            for line in content:
                match1 = re.match("(delta_rp\s*=\s*)(\d*\.?\d+)",line)
                match2 = re.match("(delta_beta\s*=\s*)(\d*\.?\d+)",line)
                match3 = re.match("(delta_alpha\s*=\s*)(\d*\.?\d+)",line)
                match4 = re.match("(alphaMIN\s*=\s*)(\d*\.?\d+)",line)
                match5 = re.match("(alphaMAX\s*=\s*)(\d*\.?\d+)",line)
                match6 = re.match("(betaMIN\s*=\s*)(\d*\.?\d+)",line)
                match7 = re.match("(betaMAX\s*=\s*)(\d*\.?\d+)",line)
                match8 = re.match("(rpMAX\s*=\s*)(\d*\.?\d+)",line)
                match9 = re.match("(rpMIN\s*=\s*)(\d*\.?\d+)",line)
                if match1:
                    #OPTIONAL: OVERWRITING DEFAULT
                    try:
                        self.r_increment =float(match1.group(2))
                        print("delta_rp=",float(match1.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                if match2:
                    #OPTIONAL: OVERWRITING DEFAULT
                    try:
                        self.delta_beta =float(match2.group(2))
                        print("delta_beta=",float(match2.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                if match3:
                    #OPTIONAL: OVERWRITING DEFAULT
                    try:
                        self.delta_alpha =float(match3.group(2))
                        print("delta_alpha=",float(match3.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail

                if match4:
                    gotMatch[0]=True
                    try:
                        self.alphamin = float(match4.group(2))
                        # print("alphaMIN=",float(match4.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                if match5:
                    gotMatch[1]=True
                    try:
                        self.alphamax = float(match5.group(2))
                        # print("alphaMAX=",float(match5.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                if match6:
                    gotMatch[2]=True
                    try:
                        self.betamin = float(match6.group(2))
                        # print("betaMIN=",float(match6.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                if match7:
                    gotMatch[3]=True
                    try:
                        self.betamax = float(match7.group(2))
                        # print("betaMAX=",float(match7.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                if match8:
                    gotMatch[4]=True
                    try:
                        self.rpMAX = float(match8.group(2))
                        # print("rpMAX=",float(match8.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                if match9:
                    gotMatch[5]=True
                    try:
                        self.rpMIN = float(match9.group(2))
                        # print("rpMIN=",float(match9.group(2)))
                    except ValueError as info:
                        err.info = str(type(info))+str(info.args) 
                        err.value =2
                        return err,-1,-1,fail
                    if(self.rpMIN<rmin):
                        err.value = 1
                        err.info = "Minimum radius in the series cannot be smaller than the minimal radius for a pocket exit, "+str(rmin)+"\nSetting rpMIN =" +str(rmin)
                        self.rpMIN = rmin
            if (not gotMatch.all()):
                try:
                    raise EOFError("A compulsory field for the operating mode chosen is not present in the input file")
                except EOFError as info:
                    err.info = str(type(info)) +str(info.args)
                    err.value = 2
                return err,-1,-1,fail

            if((self.betamin>self.betamax) or (self.alphamin > self.alphamax)or (self.rpMIN>self.rpMAX)):
                err.info = "Minimum value in range must be smaller than maximum one!\n"
                err.value = 2
                return err,-1,-1,fail

            alphas = crange(self.alphamin,self.alphamax,self.delta_alpha)
            betas = crange(self.betamin,self.betamax,self.delta_beta)
            radii = crange(self.rpMIN,self.rpMAX,self.r_increment)

            return err,Analysis,Test,(alphas,betas,radii)
            