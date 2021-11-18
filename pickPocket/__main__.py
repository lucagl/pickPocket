############ Clustering of SES probes for pocket detection and  reconstruction ############
#Author:								Luca Gagliardi 
#Copiright:					    © 2020 -2021  Istituto Italiano di Tecnologia   
############################################################### 
#PORTABLE VERSION 1.0
#PORTABLE VERSION 1.1:
#   - User options
#   - Limit with a threshold on score or a maximum number the number of pockets displayed
#   - Use a volume classifier instead of IF which shoud be faster (TODO)
from pickPocket import *
from pickPocket import global_module
import sys
import os
import subprocess

import getopt #to manage user options..

inlineCall = False
default_keep = 10
default_rankingThreshold = 0.56
argv = sys.argv[1:]

# print(argv)

rankingThreshold = default_rankingThreshold
keep = default_keep
try:
    opts, args = getopt.getopt(argv,"h",["range=","maxScore=","help"])
except getopt.GetoptError:
    print ('<<ERROR>> Uncorrect formatting of options or unavaible option')
    sys.exit(2)
print('options:',opts)
print('arguments:',args)
for opt, arg in opts:
    if opt in ["-h","--help"]:
        print('Usage:\npython3 -m pickPocket <options> <structureName>\n The structure must be in pqr format.\nOptions:')
        print('--range=<max number of pockets in ranking>: Default = %d'%default_keep)
        print('--maxScore=<(0 ,1]>: Default =%.2f, a smaller maximum score reduces the number of returned pockets'%default_rankingThreshold)
        input('\n')
        sys.exit()
    elif opt == '--range':
        keep = int(arg)
        print("OPTION: max ranked pockets= %d"%keep)
    elif opt == '--maxScore':
        rankingThreshold = float(arg)
        print("OPTION: max score Isolation Forest= %.2f"%rankingThreshold)

# input()

# INITIALIZATIONS #
### Create temp folder (if not existing)###
runPath = 'temp'
if not os.path.exists(runPath):
    os.makedirs(runPath)
# print(global_module.pathTo_NS_ex+'/*')
subprocess.call('cp '+global_module.pathTo_NS_ex+'* '+runPath+"/", shell=True)

if(len(args[0])>0):
    import re
    inputFile  = args[0] #str(sys.argv[1])
    match = re.match('([\w]*)',inputFile) #accepts both <pqrname> and <pqrname.pqr>
    inputFile = match.group(1)
    # print('\nInput structure:',inputFile)
    inlineCall = True
    proteinFile=inputFile
else:
    # sys.exit("CANNOT READ INPUT FILE")
    pass

errFile = open("errorLog.txt",'w+')
errFile.write("## ERROR LOG FILE ##\n")

logFile = open("logFile.txt", 'w')
logFile.write("******************************** LOGFILE ********************************** \n")

print()
main_crono =Crono()

t = main_crono.init()
logFile.write(t)
print(t)


## READING AND SETTING INPUTS
if(inlineCall):
    isAnalysis=False
    config = ReadConfig()
    try:
        confFile=open("config.txt",'r')
        err=config.get(confFile)
    except FileNotFoundError:
        print("Config file not found, using defaut parameters")
        logFile.write("\nConfig file not found, using defaut parameters")
        err=Error()
        err.info = "Config file not found, using defaut parameters"
        err.value = 1
    err.handle(errFile)
    alpha = config.alpha
    beta = config.beta
    rp_max = config.rp_max
    err=initFolders(inputFile)
    err.handle(errFile)
    
else:
    _input = ReadInput()
    try:
        inFile= open("input.prm",'r')
    except FileNotFoundError as info:
        print("Input file not found. You can provide pqr filename inline for simple run.\n Aborting.")
        err = Error()
        err.info=(str(type(info))+str(info.args))
        err.value=2
        err.write(errFile)
        exit()
        
    err,isAnalysis,isTest,(alphas,betas,radii) = _input.get(inFile)
    err.handle(errFile)
    # print(isAnalysis,isTest)
    if((isAnalysis==False) or (isTest ==False)):
        proteinFile=_input.getStructureIterator(single=True)
        alpha=alphas
        beta=betas
        rp_max = radii 
    else:
        print("not supported")






logFile.write("Protein structure name= "+proteinFile+"\n")
logFile.write("Clustering parameters are:\n alpha= %.1f beta= %.1f \n Maximum probe radius= %.1f, minimum probe radius= %.1f, probe increment= %.2f" 
% (alpha,beta,rp_max,global_module.R_WATER,global_module.delta_rp))
print("Clustering parameters are:\n alpha= %.1f beta= %.1f \n Maximum probe radius= %.1f, minimum probe radius= %.1f, probe increment= %.2f\n" 
% (alpha,beta,rp_max,global_module.R_WATER,global_module.delta_rp))
print("Protein structure name= "+proteinFile+"\n")

clustering = NS_clustering()
err = clustering.init(proteinFile)

err.handle(errFile)

err,pList=clustering.build(alpha,beta,rpMAX = rp_max)


err.handle(errFile)


#also performs score and size filtering 
if(pList):
    # err,pList=clustering.printInfo_old(saveSpheres = True,largeP=global_module.largeP)
    err,pList_ranked = clustering.printInfo(saveSpheres=True,keep=keep,rankingThreshold = rankingThreshold)
    logFile.write("\n Detailed info in protein output file("+global_module.pdbFolder_path+") and status.txt file("+global_module.runFolder_path)
    err.handle(errFile)
else:
    print("NO pockets found with current parameters!")
if(global_module.accTriang):
    clustering.VMD_accTriang()

logFile.write("\n\n Number of (large) pockets found: "+str(len(pList)) + " of which %d were kept in ranking"%keep)


t= main_crono.get()
print("\n ---------------- FINISHED -------------- \n")
print(t)  

logFile.write("\n ----------- FINISHED ---------- \n")
logFile.write("\n\n"+t)

n_warnings = Error.n_errors
# print(n_warnings)
if(n_warnings>0):
    logFile.write("\n\n <INFO> "+str(Error.n_errors)+" Warnings were produced.")
err.handle(errFile)
errFile.close()
logFile.close()


##################
