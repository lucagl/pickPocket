############ Clustering of SES probes for pocket detection and  reconstruction ############
#Author:								Luca Gagliardi 
#Copiright:					    © 2020 -2021  Istituto Italiano di Tecnologia   
############################################################### 

# VERSION 3.1: Stats of ranking with respect to clustering parameters (strong requirement: a subpocket hits if is ranked among 3 topmost)
# VERSION 3: ML models (isolation forsest) can be used for ranking. First prototype for direct use
# VERSION 2.2: Volume, Area and triangulation of pockets. Minor: complete feature extraction.
# VERSION 2.1: Several optimizations. Reduced computation time of about 50%.
# VERSION 2.0: Main novelty: Major architecture change. Analysis mode and single run mode. 
#              Analysis mode allows the study of clustering parameters. 
#              Meanwhile features of succesfull pockets are gathered in a binary file.
# PREVIOUS:    Proof of concept and built of core functionalities: main code = main_light.py

#NOTES TO REMEMBER FOR PAPER/PRESENTATIONS:
# - subpockets reweighted by master pocket score. So if subpocket does not belong to good master pocket penasized.
#--> not big deal considering that in any case we are using large forest for the main evaluation... BUT this should be stated explicitly..
# - Be coherent in deployent and analysis --> not sure criteria to discard subs make sense if not tested on statistics
# - Interessante discutere quante volte beccano le subs.. --> motiva il fatto di avere pockets più pulite e circoscritte
# - IF trainata su VS = 0.2 anche se testato con criterio VS = 0.5 --> smooth the decision boundaries
# - Subpockets stat on OS,VS and volume only if among top 3 substitute master if they hit..
# - Important: Also Chemistry perform more poorly if giving all pockets.. --> Larger chemical environment more predictive for ligand binding
# - Important: never tested sistematically if using small forest and or average between small and large for subs better (there are some clues)
# - Think about other argumets for using large F for geometry.. 
# - Redo plot hit performance (before ranking with no Peptides..)  
#TODO **LAST**
#   - clean clean clean!! --> stick to a single ranking method.
#                            Discard distinction between density/no density also in deployment stage..
#   - copy main with many test on a dedicated file that can be git committed separately..
#   - --> simplified version of NS for clustering runs and normal for volume calculations <--

# TODO USER:
#   - Give score based on IF --> "druggability score"
#   - Give separately chemical and geometrical scores: color somehow if a pocket is strong geometrically speaking vs chemically speaking? Give separately top 3 geometrically and chemically?
#TODO CURRENT:
# CLEAN UNUSED STuFF (for instance in train_classifier..
# # --> REDUNDANT AND SLOW TO COMPUTE TWICE THE POCKET TRIANG: 1 for volume extraction 1 for final triangulation..
# need to produce all triang and save them with current indexing. Then remove whatever not classified and then rename..

# From user perspective, think how to handle scenarios in which both subpocket and master pocket are expressed in the ranking.
# For analysis of performance is not relevant since I'm happy enough a pocket hits.
# Ideas: - keep track of pockets already expressed in ranking. If subpocket higher in ranking signal it:
#  --> point to master pocket if is in top 10; Write it when dislosing subpockets of master pocket 
# In general I need to write info in a later stage

# - It could be interesting to save the average of OS and VS of hitting pockets in the ranking as an extra selection criterion
# - Change input reading, needs new info for testing and maybe to discard info used for old printing method..
# - Load ML model at the beginning to avoid overhangs later
# - Option to filter ranking from small pockets (maybe based on volume rather than ranking..). If high in ranking their position is freed
#   still, it seems to me that count threshold is more functional for us.
# Hard code surfaceConfiguration.prm NS file so that the code produces it upon call
# --> Accurate triangulation and grid from input for single run mode
#  IMPORTANT: rmin user defined ?? --> could be incoherent with ML training

#OBSERVATIONS ON MAJOR INCONSISTENCIES:
# - Mouths feature is meaningful only for pockets with subpockets. A pocket with no subpockets (or of course a subpocket itself)
#   has no other mouths except the defining exit



#RECENT CHANGES:
#   Features gathered only on succesful hits --> reduce amount of data saved
#   Volume extraction implemented 
# Triangulation via NS implemented

########### NOTES ###########
                # D:= Conventional threshold distance. Default = 4 Amstrongs
                # OS := Overlap Score (ligand hit evaluation) 
                #       = number ligand's heavy atoms within D from protein pocket's atoms divided total number of ligand's heavy atoms
                # VS := Match Volume Score (volume coverage evaluation) 
                #        = number of protein atoms of the pocket within D from ligands atoms divided by total number of protein pocket's atom
#-------------------
#   - The main assumption is that our conventional definition of pockets holds --> we are NOT optimizing gathering (postprocessing) parameters.
#   - There can be 1 hit per structure (structure = protein + single ligand). We keep the best match (see below).
#   - DEF. Math= 20% OS and 50% VS (rounded to 1st decimal)

########## MAIN OUTPUTS #########
# A. Single run mode:
#       Description of pockets found (with heuristic description also). Save the pockets sphere on pqr file.
# B. Analysis mode: TO UPDATE! <--
#   1.  Local hit_file: alpha   beta    rpMAX   OverlapScore    VolumeScore
#   2.  Summary hit_file: alpha beta    rpMax   Average OS      Average VS  Hit_stat
#   3.  Feature raw data containing a map between score of succesfull pockets, persistence and bottlenecks is saved in a binary pickle file.
#  Methodological Note:
#       Average scores are obtained averaging only over succesful pocket-ligand matches. Hit_stat is the success rate of a set of parameters.
#       If more than 1 pocket x structure matches we keep the one with best OS (does NOT imply best VS).
#       No match (0,0) score (no match = no sufficient OS and VS -- see above)

########### ERROR HANDLING ###################
#err.value = 2 must cause exit
#err.value = 1 is an handled exception
# We try to minimize errors that cause crash in analysis mode. Structure which cause error will be skipped and not contribute in averages.

## TODO IMPROVEMENTS ##
# Suppress lightMode: unused.. OK
# Change in NS input file can be done more efficiently. Now is redundant, most of time you just need to change one line without reach the end of file..(NOT IMPORTANT)
# Modify NS to stop where needed (only for clustering process, not pocket triangulation..) --> IMPORTANT FOR EFFICIENCY.. 

#Don't like to load protein object into pocket for matchscore.. Change architecture?

#  Invert loop protein ligands so that ligands loop is the nested one?

#NOTES ON SELF INTERSECTION:
# If we use probes relative to surface patches with no self int, we loose by construction the possibility of bottlenecks.
# However, this one day could be interesting when looking at protein protein interactions, where such small features do not matter..
# Not sure, think about. Of course there are enormus performance beneficts

#NOTES ON I THINK SMALL PROBLEM (OR MAYBE DESIDERABLE BEHAVIOR..):
# Regarding persistence of subpockets.. 
#The unfiltered id_shifts from subpocket perspective, could not correspond (only for topmost mouths) 
# to actual depth and count, of the complete filtered set of mouths known from the parent pocket.
# Therefore, shen selecting persistence comparing to master pocket filtered persistence we might loose info:
#Example alpha=0.4, bta=0.5 6gj6 pocket0(largest): 3rd subpocket has unfiltered persistence containing all the series leading to topmost mouth
# and other 2 entries which are not found in the filtered array. This because for sure there is a further aggregation in the series at a parent node.
# Since we select subpockets from the (filtered) mouth-list that was probaly related to a larger subpocket which is dropped.
#So this shows that selected cluster can contain pieces of another cluster (pocket) because there is no guarantees the clustering process follosw an
#order which prioritize radial shifts. This is due to the fact that before each radial shift there is a lateral aggregation..

#NOTES FEATURES DATA:
# I expect in general to be redundant. Actually what will happen is that the best score (small parameters)
# will correspond to a subpocket match of broad parameters. 
#--> Consider gathering features on matchins subpockets.

#NOTES ON CLUSTERING AND INTERPLAY BETWEEN ALPHA AND BETA:
# It's important to keep in mind that tag=3 is possible only between a larger probe which is isolated(*) and a cluster or a singleton of small probes.
# This also is why interplay alpha - beta is complex. It might happen that a larger alpha hinders the possibility of building a series of tag=3.
# (*) However "pyramidal case" can happen due to the sorting of distances
# DEVEL: progressive alphas and betas or stop positive progression with larger probes

#----------
# NEW VERSION:
#
# run--> 2 ways: 
#     1. Euristic class (as now, but rationalizing tags) 
#     2. Test from trained model (option tree and SVDD for different features - in trees feat should be more transparent)
# analyse --> 2ways:
#     1. Usual
#     2. Test for all parameters if tested pockets fall within top 10 or top 3 of hitting pockets. 
#        - Test both global ranking and large pocket ranker. For global use the whole range (no in out)
#        - When using the large pocket ranker, test if subpockets (ranked according to global ranker) hit within first 3.
#        - Test a mixture of the 2, large ranker as an in/out tester, and remaining places are filled by the global ranker , 
#           again for subpockets of high ranked pockets use the above approach.
#        - ATTENTION: a. Need to skip subpocket if master pocket higher in  --> this is strictly true
#                     b. Skip parent pocket if single subpocket higher in ranking
#                     c. Skip single subpocket if parent higher in ranking (?, or treat it as for thes scenario with many subpockets..)





import sys
import os


from standaloneWrapper import NS_clustering
from functions import readLigands,saveResSimple

import global_module
from global_module import ReadInput
from global_module import ReadConfig
from global_module import Error
from global_module import initFolders
from global_module import Crono

import numpy as np
np.seterr( invalid='ignore') #avoids warning when no match causes 0/0 division when gathering stats


makeInfo = True
saveLocal = False
saveP = True  # save res and number of hitting pocket.. CAREFUL use for single clustering parameter!

excludePeptide = False
VS_threshold = 0.2

amino = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','HID','HIE','HIP','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}

class ContinueI(Exception):
    pass

################## MAIN ###############################
def main():
    inlineCall = False
    # INITIALIZATIONS #
    if(len(sys.argv)>1):
        import re
        inputFile  = str(sys.argv[1])
        match = re.match('([\w]*)',inputFile) #accepts both <pqrname> and <pqrname.pqr>
        inputFile = match.group(1)
        print(inputFile)
        # print("Single run mode on passed pqr file")
        inlineCall = True
        proteinFile=inputFile
    else:
        # sys.exit("CANNOT READ INPUT FILE")
        pass

    errFile = open("errorLog.txt",'w+')
    errFile.write("## ERROR LOG FILE ##\n")

    logFile = open("logFile.txt", 'w')
    logFile.write("******************************** LOGFILE ********************************** \n")

    main_crono =Crono()
    meso_crono = Crono()
    local_crono = Crono()
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
        if((isAnalysis==False) and (isTest ==False)):
            proteinFile=_input.getStructureIterator(single=True)
            alpha=alphas
            beta=betas
            rp_max = radii 
        else:
            structures = _input.getStructureIterator()
            
            # if(lightMode):
            #     logFile.write("\n **Light mode selected. Features data won't be produced\n")
            #     print("** Light mode selected. Features data won't be produced\n")

            # if(onlySub):
            #     print("\n ** CAREFULL: SUBPOCKETS MODE: Subpockets are considered as independent objects for scoring purposes and feature production.\n")
            #     logFile.write("\n ** CAREFULL: SUBPOCKETS MODE: Subpockets are considered as independent objects for scoring purposes and feature production.\n")
            print("alphas array=",alphas)
            print("betas array=",betas)
            print("radii array =",radii)
            print(structures)
            n_structures = len(structures)

            logFile.write("\nNumber of structures : %d \n Number of parameters to check x structure: %d = n_alpha=%d X n_beta=%d X n_radii=%d\n" 
            %(len(structures),alphas.size * betas.size * radii.size,alphas.size,betas.size,radii.size))
            print("\n ++Number of structures : %d \n Number of parameters to check x structure: %d = n_alpha=%d X n_beta=%d X n_radii=%d ++ \n" 
            %(len(structures),alphas.size * betas.size * radii.size,alphas.size,betas.size,radii.size))
            logFile.write("\nAlphas: " )
            logFile.writelines(str(alphas))
            logFile.write("\nBetas: " )
            logFile.writelines(str(betas))
            logFile.write("\nRadii: " )
            logFile.writelines(str(radii))
            logFile.write("\nStructures analysed: " )
            for s in structures:
                    logFile.write("\n")
                    for k, v in s.items():
                        logFile.write(str(k) + ' -->'+ str(v)+'\t')
                

    ############
    # lightMode=global_module.lightMode

    ######## RUNNING CLUSTERING ALGOTITHM
    print("\n\n ** STARTING CORE ALGORITHM **\n")
    logFile.write("\n\n ** STARTING CORE ALGORITHM ** \n")

    if(isTest):
        from train_classifier import Scoring,getFeat,getFeatAlt,getFeatNoDensity,getFeatOnlyDensity
        

        #   *********** LOAD TRAINED MODEL ****************

        from functions import save_rankingStats
        print("\n++++TEST MODE: assessing performance of predictor on structures +++++\n")#generic, could be training as new samples
        logFile.write("\n++++TEST MODE: assessing performance of predictor on structures +++++\n")
        # +++++++++++++++++++ Load trained classifier +++++++++++++++++++++
        
        scoreIF_noDensity = Scoring()
        # scoreIF_onlyDensity = Scoring()
        
        model1="IF10_noPeptides" #<-- Put here the trained model you want to use..
        # model2="IF10_onlyDensity" # Hydro density replaces hydro count
      
        
        names = ["rankStats_IF_geomChem.out","rankStats_IF_onlyGeom.out","rankStats_IF_onlyChem.out"]

        err = scoreIF_noDensity.load('../'+model1,modelType=1,unique=False) #change here and below to test different classifiers..
        # err = scoreIF_onlyDensity.load(model2,modelType=1,unique=False)

        
        err.handle(errFile)

        n_models = 3 # Ranking modes..

        hitTop3 = np.zeros((alphas.size,betas.size,radii.size,n_models))
        hitTop10 = np.zeros((alphas.size,betas.size,radii.size,n_models))
        hitTopWithSub = np.zeros((alphas.size,betas.size,radii.size,n_models))
        hitTopOnlySub = np.zeros((alphas.size,betas.size,radii.size,n_models))

        hitTop1Sub = np.zeros((alphas.size,betas.size,radii.size,n_models))

        allHit = np.zeros((alphas.size,betas.size,radii.size,n_models))
        OS_log = np.zeros((alphas.size,betas.size,radii.size,n_models))
        VS_log = np.zeros((alphas.size,betas.size,radii.size,n_models))
        volume_log = np.zeros((alphas.size,betas.size,radii.size,n_models))

        norm = np.zeros((alphas.size,betas.size,radii.size,n_models))
        nAnalysed = 0

        # nohitMap =[{}] * n_models
        nohitMap =[]
        for i in range(n_models):
            nohitMap.append({})
        
        ff = [open("noTopHit_geomChem_"+model1+".out",'w'),open("noTopHit_onlyGeom_"+model1+".out",'w'),open("noTopHit_onlyChem_"+model1+".out",'w')]


        #LOOP OVER STRUCTURES
        outFile = open("output_succesfullP.out",'w')
        outFile.write("# ligandStruct\tpname\tOS\tVS\n")

        skipped_structures = 0
        clustering = NS_clustering()
        for s in range(n_structures):
            print("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n" )
            logFile.write("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n")
            
            proteinFile = structures[s]['pqr']
            ligands = structures[s]['ligands']
            
            isPeptide = []
            for ln in ligands:
                if (bool(set(ln.split('_')) & amino)):
                    isPeptide.append(True)
                else:
                    isPeptide.append(False)
            # print("Loading ligand(s) coordinates")
            ligands = [(ln,isP) for ln,isP in zip(ligands,isPeptide)]

            
            # print("Loading ligand(s) coordinates")
            ligands_coord=[]

            # INITIALIZATION : loading structure data...
            err = clustering.init(structure_name=proteinFile)
            if(err.value==2):
                logFile.write("\n A MAJOR WARNING WAS PRODUCED\n")
                err.info = err.info + "\nSkipping the structure. Cannot load " + proteinFile
                err.value = 1
                err.handle(errFile)
                skipped_structures+=1
                continue
            err.handle(errFile) 

            #Prepare container with ligand coordinates
            for ligand_name in ligands:
                #NEW OPTION --> exclude Peptide
                if((ligand_name[1] == True) and excludePeptide):
                    print("SKIPPING PEPTIDE "+ligand_name[0])
                    logFile.write("\n SKIPPING PEPTIDE "+ligand_name[0])
                    err.info = err.info + "\n" + "Skipping "+ligand_name[0]+" of structure "+proteinFile+": PEPTIDE"
                    err.value = 1
                    err.handle(errFile)
                    continue
                try:
                    coord = readLigands(global_module.pdbFolder_path+ligand_name[0],proteinCoord=clustering.get_protein().atoms)
                    if (len(coord)==0):
                        print("Skipping "+ligand_name[0]+" of structure "+proteinFile+ " : No ligands heavy atoms within 5A from protein")
                        err.value = 1
                        err.info = "Skipping "+ligand_name[0]+" of structure "+proteinFile + " : No ligands heavy atoms within 5A from protein!"
                        err.handle(errFile)

                        continue
                    ligands_coord.append({"name":ligand_name[0],"coord":coord, "isPeptide":ligand_name[1]})
                    # if(saveLocal):
                    #     fp = open(global_module.pdbFolder_path+"hitfile_"+proteinFile+"_"+ligand_name+".out",'w')
                    #     fp.write("# ** Hit file for "+ proteinFile+" with ligand "+ligand_name+ "**")
                    #     fp.write('\n#'+ Crono().init())
                    #     fp.write("\n# alpha\tbeta\trp_max\tOverlapScore\tVolumeOverlap\tNumber of pockets found")
                    #     localHitFile.append(fp)
                except NameError:
                    err.value = 1
                    err.info = "Ligand " + ligand_name[0] +" could not be loaded. \n Skipping, it won't be considered among available comparesons. "
                    err.handle(errFile)
                    print("SKIPPING: Ligand  "+ligand_name[0]+ " could not be loaded")
                    continue
            if ((not ligands_coord)and (saveP==False)):
                err.value = 1 
                err.info = "Skipping the structure " + proteinFile+". No valid ligands found."
                err.handle(errFile)
                print("SKIPPING the structure: no valid ligands found")
                logFile.write("\n SKIPPING: no  valid ligands found..")
                skipped_structures+=1
                #skip current iteration: go to next structure
                continue

            
            
            # ++++++++++++++++++ LOOP OVER ALL PARAMETERS ++++++++++++++++++
            meso_crono.init()
            
            for i,alpha in enumerate(alphas):
                for j,beta in enumerate(betas):
                    for k,rp_max in enumerate(radii):
                        # local_crono.init()
                        #RUN CLUSTERING ALORITHM
                        # print("PARAMETERS: alpha= %.1f; beta = %.1f"%(alpha,beta))
                        try:
                            err,pList=clustering.build(alpha,beta,rpMAX = rp_max)
                        except Exception:
                            err.value = 2
                            print("\n Critical problem 1 ")
                            err.info = err.info + "An unhandled exception was produced\n "#Skipping alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) +"of current protein="+proteinFile
                            logFile.write("\n A CRITICAL WARNING WAS PRODUCED\n")
                            err.handle(errFile)
                        if(err.value==1):
                            err.info= err.info + "\n Skipping current triplet due to failure in clusters construsction \n Skipping alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) +"of current protein="+proteinFile
                            # err.value = 1 #prevents exit, in this case we want to skip
                            err.handle(errFile)
                            #NOT EASY TO ACCOUNT FOR THIS WHEN AVERAGING MANY RUNS.. BETTER FULL SKIP OF THE STRUCTURE?
                            continue 
                        elif(err.value==2):
                            print("\n Critical problem 2")
                            err.info = err.info + "An unhandled exception was produced\n"#Skipping alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) +"of current protein="+proteinFile
                            logFile.write("\n A CRITICAL WARNING WAS PRODUCED\n")
                            err.handle(errFile)
                            break
                        # ** ANALYSE CANDIDATE POCKETS FOUND **
                        
                        
                        featListGeom,featListChem,map_direct,map_reverse = getFeatNoDensity(pList,clustering.get_protein().resMap)
                        
                        
                        scoreIF_noDensity.resetRank()
                        rankedIndexesIF,rankedIndexesSubIF,_numericalScoreIF,_numericalScoreSubIF = scoreIF_noDensity.getRanking(featListGeom,featListChem)
                        rankIF_geomChem = (rankedIndexesIF,rankedIndexesSubIF)

                        # rankedIndexesIF,rankedIndexesSubIF,_numericalScoreIF,_numericalScoreSubIF = scoreIF_noDensity.getRankingOnlyChem(featListChem)
                        # rankIF_onlyChem = (rankedIndexesIF,rankedIndexesSubIF)

                        # rankedIndexesIF,rankedIndexesSubIF,_numericalScoreIF,_numericalScoreSubIF = scoreIF_noDensity.getRankingOnlyGeom(featListGeom)
                        # rankIF_onlyGeom = (rankedIndexesIF,rankedIndexesSubIF)

                        

                    
                        ranks = [rankIF_geomChem]
                        
                        save_path = 'output/'+structures[s]['pqr']+'_pockets'
                        m=0


                        for ln,l in enumerate(ligands_coord):
                            # print(l['name'])
                            top3 = np.zeros(n_models)
                            top10 = np.zeros(n_models)
                            topwithSub =np.zeros(n_models)
                            topOnlySub = np.zeros(n_models)

                            top1Sub = np.zeros(n_models)

                            OS_kept = np.zeros(n_models)
                            VS_kept = np.zeros(n_models)

                            #NEW
                            volume = np.zeros(n_models)

                           

                            rankMain = ranks[m][0]
                            rankSub =  ranks[m][1]# For some ranking modes it coincides with the above..
                            #ligand-structure loop
                            r=0 # rank position
                            r_in=0 # running over ranked indexes array
                            selfContained = set()
                            
                            while ((r<10) and (r_in < rankMain.size)):
                                pi,si = map_direct[rankMain[r_in]] #pocket and subpocket original index

                                if(si is not None):
                                    # ------------------- SUBPOCKET  (TREATED AS MASTER FROM USER PERSPECTIVE)--------------------
                                    if(pi in selfContained):
                                        #going to next element (positions do not progress in the ranking) since sub contained in master pocket higher in ranking
                                        r_in+=1
                                        continue # r index does not advance

                                    pocket = pList[pi]['subpockets'][si]['node'] #IS A SUBPOCKET
                                    pocket.load_protein(clustering.get_protein())
                                    # Question: evaluate goodnes of subpocket based only on volume score? (much more severe..)
                                    gotHit,OS,VS = pocket.matchScore(l['coord'],coverageTh=VS_threshold)
                                    if gotHit:
                                        OS_kept[m] = OS
                                        VS_kept[m] = VS
                                        volume[m],_A,_err = pocket.volume()

                                        top10[m] = 1
                                        # This counter is needed only to distingish scenarios where master pocket does not hit or the contrary
                                        topwithSub[m] = 1 
                                        if(r<3):#within top3
                                            top3[m] = 1
                                        if(saveP and m==0):
                                            if not os.path.exists(save_path):
                                                os.makedirs(save_path)
                                            saveResSimple(r,save_path,pocket,clustering.get_protein().resMap)
                                            outFile.write(proteinFile+'--'+l['name']+'\tp'+str(r)+'\t%.2f\t%.2f\n'%(100*OS,100*VS))
                                        break
                                    # removing redundancy on single subpocket. If single subpocket higher ranked, I consider it redundant with master one 
                                    if(len(pList[pi]['subpockets'])==1):
                                    #Single subpocket--> master pocket in black list
                                        selfContained.add(pi)

                                else:
                                    # --------- PARENT POCKET (or with no subpockets) --------------------
                                    if(pi in selfContained): #SKIP PARENT POCKET OF A SINGLE SUBPOCKET ALREADY EXPRESSED IN THE RANKING
                                        r_in+=1
                                        continue
                                    selfContained.add(pi) #to filter out subpockets already expressed by the master pocket

                                    pocket = pList[pi]['node']
                                    pocket.load_protein(clustering.get_protein())
                                    gotHit,OS,VS = pocket.matchScore(l['coord'],coverageTh=VS_threshold)
                                    subs = pList[pi]['subpockets']
                                    n_subs = len(subs)
                                    if gotHit:
                                        OS_kept[m] = OS
                                        VS_kept[m] = VS
                                        volume[m],_A,_err = pocket.volume()

                                        top10[m] = 1
                                        if(n_subs == 0):
                                            # If no subs, I don't want to penalize sub hit or it would be unreadable
                                            topwithSub[m] = 1
                                        if(r<3):#within top3
                                            top3[m] = 1
                                        if(saveP and m==0):
                                            if not os.path.exists(save_path):
                                                os.makedirs(save_path)
                                            saveResSimple(r,save_path,pocket,clustering.get_protein().resMap)
                                            outFile.write(proteinFile+'--'+l['name']+'\tp'+str(r)+'\t%.2f\t%.2f\n'%(100*OS,100*VS))
                                            
                                    internal_rank=[] # container for score of subpockets analysed 
                                    
                                    indS = []# just to avoid an extra "if" clause
                                    # if(n_subs>0):
                                    for sub_i in range(n_subs):
                                        internal_rank.append(np.where(rankSub==map_reverse[(pi,sub_i)])[0][0])
                                        indS = np.argsort(internal_rank) #relative rank among subpockets
                                    for ns,sub_i in enumerate(indS):
                                        subpocket = subs[sub_i]['node']
                                        subpocket.load_protein(clustering.get_protein())
                                        gotHitsub,OS,VS = subpocket.matchScore(l['coord'],coverageTh=VS_threshold)
                                        if (gotHitsub and ns < 3):
                                            #IMPORTANT : I'm counting it only if subpocket among top3 and master pocket didn't hit TODO
                                            #OS AND VS OVERWRITTEN BY SUBPOCKET --> MAYBE TOO MUCH OF CHEATING ?
                                            # Do the same for volumes?
                                            topwithSub[m] = 1
                                            if ns==0:
                                                top1Sub[m] = 1
                                            if(gotHit):
                                                pass
                                            else:
                                                topOnlySub[m] = 1
                                                top10[m] = 1 
                                                if(r<3):
                                                    top3[m] = 1
                                            OS_kept[m] = OS
                                            VS_kept[m] = VS
                                            volume[m],_A,err = subpocket.volume() 
                                            
                                            if(saveP and m==0):
                                                if not os.path.exists(save_path+'/sub'):
                                                    os.makedirs(save_path+'/sub')
                                                saveResSimple(r,save_path,subpocket,clustering.get_protein().resMap,nsub=sub_i)
                                                outFile.write(proteinFile+'--'+l['name']+'\tp'+str(r)+'_sub'+str(ns)+'\t%.2f\t%.2f\n'%(100*OS,100*VS))
                                            break
                                    if gotHit:
                                        #OF MASTER POCKET
                                        break
                                r+=1
                                r_in+=1
                           
                                if((r==10) or  (r_in == rankMain.size) ):
                                    # print('here')
                                    # print(structures[s]['pqr'],structures[s]['ligands'])
                                    # print(m,"failed")
                                    #not whithin top 10
                                    # print(nohitMap[m])
                                    if((alpha,beta,rp_max)in nohitMap[m]):
                                        # print('here1')
                                        nohitMap[m][alpha,beta,rp_max].append(structures[s]['pqr']+'_'+structures[s]['ligands'][ln])
                                    else:
                                        # print('here2')
                                        nohitMap[m][alpha,beta,rp_max]=[structures[s]['pqr']+'_'+structures[s]['ligands'][ln]]
                                    # print(nohitMap[m])
                            
                            # WITHIN LIGAND LOOP
                            hitTop3[i,j,k,:] +=top3
                            hitTop10[i,j,k,:] +=top10
                            hitTopWithSub[i,j,k,:] += topwithSub
                            hitTopOnlySub[i,j,k,:] += topOnlySub

                            hitTop1Sub[i,j,k,:] += top1Sub
                            
                            

                            allHit[i,j,k,:] += np.bitwise_or(topwithSub.astype(int),top10.astype(int)) #int required by bitwise_or..
                            OS_log[i,j,k,:]+=OS_kept
                            VS_log[i,j,k,:]+=VS_kept
                            volume_log [i,j,k,:]+=volume
                            norm[i,j,k,:] +=1
                        
            #OUT OF ALL PARAMETER LOOPS
            nAnalysed += len(ligands_coord) #n structure ligand pairs                
            avHitTop3 = np.nan_to_num(hitTop3/norm)
            avHitTop10 = np.nan_to_num(hitTop10/norm)
            avHitTopwithSub = np.nan_to_num(hitTopWithSub/norm)
            avHitTopOnlySub = np.nan_to_num(hitTopOnlySub/norm)

            avHitTop1Sub = np.nan_to_num(hitTop1Sub/norm)

            average_OS_log=np.nan_to_num(OS_log/allHit)#average only over success
            average_VS_log=np.nan_to_num(VS_log/allHit)
            average_volume_log = np.nan_to_num(volume_log/allHit)
            # print(norm)
            # SAVING METHOD.. 
            for m in range(n_models):
                save_rankingStats(m,alphas,betas,radii,avHitTop3,avHitTop10,avHitTopwithSub,avHitTopOnlySub,avHitTop1Sub,average_OS_log,average_VS_log,average_volume_log,nAnalysed,names,date=Crono().init())
                
            # FLUSH BUFFER 
            # print(s)
            advancement =(100*np.round((s+1)/n_structures,3),s+1-skipped_structures,n_structures)
            print("ADVANCEMENT: %d %% of structures"%advancement[0])
            t=meso_crono.get()
            print('\n',t)
            logFile.write(t+"\n")
            logFile.write("ADVANCEMENT: %d %% of structures"% advancement[0])
            logFile.flush()
            errFile.flush()
        #Summary of structures-Ligand not classified into top 10
        for m in range(n_models):
            for alpha in alphas:
                for beta in betas:
                    for rp_max in radii:
                        if((alpha,beta,rp_max)in nohitMap[m]):
                            ff[m].write("\n%.2f\t%.2f\t%.2f" %(alpha,beta,rp_max))
                            ff[m].write(repr(nohitMap[m][alpha,beta,rp_max]).replace("[","\n").replace("]","").replace(", ","\n"))
            ff[m].close()
        outFile.close()
    ####################### STATS ON MANY PARAMETERS
    elif (isAnalysis): 
        from functions import getEntrance
        from functions import writeBinary
        from functions import save_avHit
        print("\n++++ANALYSIS MODE: assessing best clustering parameters +++++\n")
        logFile.write("\n+++++ ANALYSIS MODE: assessing best clustering parameters. +++++\n")
        ## ANALYSIS RESULTS CONTAINERS
        # Ligand match measures
        Hit=np.zeros((alphas.size,betas.size,radii.size))
        norm = np.zeros((alphas.size,betas.size,radii.size))
        OS_log = np.zeros((alphas.size,betas.size,radii.size))
        VS_log = np.zeros((alphas.size,betas.size,radii.size))

        Hit_noP=np.zeros((alphas.size,betas.size,radii.size))
        norm_noP= np.zeros((alphas.size,betas.size,radii.size))
        OS_log_noP = np.zeros((alphas.size,betas.size,radii.size))
        VS_log_noP = np.zeros((alphas.size,betas.size,radii.size))

        nohitMap ={}
        
        # Containers for stats on all pockets ans subpockets found (independently from match)

        processed_structures = np.zeros((alphas.size,betas.size,radii.size)) #counts processed structure x parameter (robust to skipping)

        nPockets=np.empty((alphas.size,betas.size,radii.size))
        sum_nPockets=np.zeros((alphas.size,betas.size,radii.size))
        sum_norm_nPockets=np.zeros((alphas.size,betas.size,radii.size))
        
        nSubs = np.empty((alphas.size,betas.size,radii.size))
        sum_nSubs=np.zeros((alphas.size,betas.size,radii.size))
        sum_norm_nSubs = np.zeros((alphas.size,betas.size,radii.size))
        
        nSpheres =np.empty((alphas.size,betas.size,radii.size))
        sum_nSpheres = np.zeros((alphas.size,betas.size,radii.size))
        sum_norm_nSpheres = np.zeros((alphas.size,betas.size,radii.size))

        nSpheres_xP=np.empty((alphas.size,betas.size,radii.size))
        sum_nSpheres_xP = np.zeros((alphas.size,betas.size,radii.size))
        sum_norm_nSpheres_xP = np.zeros((alphas.size,betas.size,radii.size))
        
        Volume = np.empty((alphas.size,betas.size,radii.size))
        sum_Volume = np.zeros((alphas.size,betas.size,radii.size))
        norm_Volume = np.zeros((alphas.size,betas.size,radii.size))
        #######################
        #####################

        ### ++ File containing list of failed hits per parameter ++
        ff = open("failure_list.txt",'w')
        ff.write("#alpha\tbeta\trp_max")

        # for every triplet, list of structure-ligands with no hit.

        clustering = NS_clustering()
        # if(not lightMode):
        pstream = writeBinary("features_data.pkl")
        nAnalysed = 0
        skipped_structures = 0
        nPeptides =0 
        for s in range(n_structures):
            saveP_path = 'output/'+structures[s]['pqr']+'_Pres'
            #MAIN LOOP OVER STRUCTURES
            print("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n" )
            logFile.write("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n")
            proteinFile = structures[s]['pqr']
            ligands = structures[s]['ligands']
            
            # print("ligands:", ligands)
            isPeptide = []
            for ln in ligands:
                if (bool(set(ln.split('_')) & amino)):
                    isPeptide.append(True)
                else:
                    isPeptide.append(False)
            # print("Loading ligand(s) coordinates")

            ligands = [(ln,isP) for ln,isP in zip(ligands,isPeptide)]

            # print("ligands + isPeptide:", ligands)


            ligands_coord=[]
            localHitFile=[]

            # INITIALIZATION : loading structure data...
            err = clustering.init(structure_name=proteinFile)
            if(err.value==2):
                logFile.write("\n A MAJOR WARNING WAS PRODUCED\n")
                print("SKIPPING THE STRUCTURE: check errorLog")
                err.info = err.info + "\nSkipping the structure. Cannot load " + proteinFile
                err.value = 1
                err.handle(errFile)
                skipped_structures+=1
                continue
            err.handle(errFile) 

            for ligand_name in ligands:
                # print(ligand_name)    
                try:
                    coord = readLigands(global_module.pdbFolder_path+ligand_name[0],proteinCoord=clustering.get_protein().atoms)
                    if (len(coord)==0):
                        print("Skipping "+ligand_name[0]+" of structure "+proteinFile+ " : No ligands heavy atoms within 5A from protein")
                        err.value = 1
                        err.info = "Skipping "+ligand_name[0]+" of structure "+proteinFile + " : No ligands heavy atoms within 5A from protein!"
                        err.handle(errFile)

                        continue
                    ligands_coord.append({"name":ligand_name[0],"coord":coord, "isPeptide":ligand_name[1]})
                    if(ligand_name[1]==True):
                        nPeptides+=1
                    if(saveLocal):
                        fp = open(global_module.pdbFolder_path+"hitfile_"+proteinFile+"_"+ligand_name[0]+".out",'w')
                        fp.write("# ** Hit file for "+ proteinFile+" with ligand "+ligand_name[0]+ "**")
                        fp.write('\n#'+ Crono().init())
                        fp.write("\n# alpha\tbeta\trp_max\tOverlapScore\tVolumeOverlap\tNumber of pockets found")
                        localHitFile.append(fp)
                except NameError:
                    err.value = 1
                    err.info = "Ligand " + ligand_name[0] +" could not be loaded. \n Skipping, it won't be considered among available comparesons. "
                    err.handle(errFile)
                    print("SKIPPING: Ligand  "+ligand_name[0]+ " could not be loaded")
                    continue
            if not ligands_coord:
                err.value = 1 
                err.info = "Skipping the structure " + proteinFile+". No ligands found."
                err.handle(errFile)
                print("SKIPPING the structure: no ligands found")
                logFile.write("\n SKIPPING: no ligands found..")
                skipped_structures+=1
                #skip current iteration: go to next structure
                continue
            # print(proteinFile)


            meso_crono.init()
            for i,alpha in enumerate(alphas):
                for j,beta in enumerate(betas):
                    for k,rp_max in enumerate(radii):
                        # print("\nComputing: alpha= %.2f, beta= %.2f ,rp_max = %.2f " %(alpha,beta,rp_max))
                        # logFile.write("\n\nComputing: alpha= %.1f, beta= %.1f ,rp_max = %.1f \n"%(alpha,beta,rp_max))
                        ###            **CLUSTERING STEP**
                        local_crono.init()
                        try:
                            err,pList=clustering.build(alpha,beta,rpMAX = rp_max)
                        except Exception:
                            err.value = 2
                            print("\n Critical problem 1 ")
                            err.info = err.info + "An unhandled exception was produced\n "#Skipping alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) +"of current protein="+proteinFile
                            logFile.write("\n A CRITICAL WARNING WAS PRODUCED\n")
                            err.handle(errFile)
                        if(err.value==1):
                            err.info= err.info + "\n Skipping current triplet due to failure in clusters construsction \n Skipping alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) +"of current protein="+proteinFile
                            # err.value = 1 #prevents exit, in this case we want to skip
                            err.handle(errFile)
                            #NOT EASY TO ACCOUNT FOR THIS WHEN AVERAGING MANY RUNS.. BETTER FULL SKIP OF THE STRUCTURE?
                            continue 
                        elif(err.value==2):
                            print("\n Critical problem 2")
                            err.info = err.info + "An unhandled exception was produced\n"#Skipping alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) +"of current protein="+proteinFile
                            logFile.write("\n A CRITICAL WARNING WAS PRODUCED\n")
                            err.handle(errFile)
                            break
                        # err.handle(errFile) 
                        ## ------------------------------

                        #NEW: sorting from larger to smaller ensures in the rare cases of parity that potentially "worst" VS is returned
                        #(NOTE that OS wins anyway. This might only influence outcome if OS identical)
                        pList=sorted(pList, key = lambda x: x['node'].count)[::-1]

                        #I think that the scale does not matter for accTriang = false
                        # clustering.pocketMode(grid_scale=2,grid_selfInt=3,maxProbes_selfInt=200,accTriang=False)#setup volume calculation for pocket clusters

                        # EVALUATION OF LIGANDS
                        # hit is maximum 1 per ligand-structure pair. Then I keep best score if more than 1 hit
                        try:
                            #PARAMETERS ITERATOR
                            for ln,l in enumerate(ligands_coord):
                                #LIGAND ITERATOR
                                hit = 0
                                OS_max= 0
                                VS_kept = 0
                                
                                # --> If more than one hit per ligand, keep the best according to OS score <--
                                try:
                                    volume_t = 0
                                    nspheres = 0
                                    nsubs = 0
                                    pCounter=0
                                    for pn,p in enumerate(pList):
                                        #POCKET ITERATOR
                                        nspheres += p['node'].count
                                        nsubs += len(p['subpockets']) #0 if lightMode on, since not built..
                                    
                                        volume,area,err = p['node'].NSVolume(accurate=False)

                                        # print("P"+str(pn) +" Volume= "+str(volume)+"A= "+str(area))
                                        if(err.value==1):
                                            # print("P"+str(pn) +" Volume= "+str(volume)+"A= "+str(area))
                                            # if(ln==0):
                                            #     err.info= "Structure= "+proteinFile+": Error in volume computation of pocket"+str(pn)+" for parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max)+":  "+err.info + "\nTrying with ARVO module computing volume of overlapping spheres"
                                            #     err.handle(errFile)
                                            #     print("Error in volume computation of pocket "+str(pn)+" for parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max))
                                            #     print("Trying with analytic approach..")
                                            volume,area,err = p['node'].volume()
                                            # print(volume,area)
                                            if (err.value==1):
                                                #I expect this never happens..
                                                print("SKipping pocket"+str(pn)+": Cannot compute volume. Parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max))
                                                err.info="\n"+"Structure= "+proteinFile+" Cannot compute volume of pocket"+str(pn)+"parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max)+ "\nSKIPPING pocket!\n"
                                                err.handle(errFile)
                                                continue
                                            # else:
                                            #     print("OK!")
                                            #     err.info= err.info+"\n" + "OK!"
                                            #     err.value=1#to force writing of error info..
                                            #     err.handle(errFile)

                                        volume_t += volume
                                        pCounter+=1 
                                    
                                        # if(onlySub and p['subpockets'] and (not lightMode)):
                                        # else: 
                                        p['node'].load_protein(clustering.get_protein())
                                        # try:
                                        gotHit,OS,VS = p['node'].matchScore(l['coord']) #success test performed internally. OS,VS>0 only if success 
                                        # if(not lightMode):
                                        # except Exception:
                                        #     #
                                        #     print("Structure "+proteinFile+": Cannot compare to ligand "+l['name'] +" Internal error. Skipping!")
                                        #     err.info = "Structure "+proteinFile+": Cannot compare to ligand "+l['name'] +" Internal error. Skipping!"
                                        #     err.value=1
                                        #     err.handle(errFile)
                                        #     raise ContinueI 
                                        # if(OS>0):
                                        if(gotHit):
                                            #RECOMPUTE ACCURATE VOLUME AND AREA FOR SUCCESSFUL POCKETS
                                            
                                            volume,area,err = p['node'].NSVolume(accurate=True)
                                            
                                            #If it failed with NS and did not with ARVO, an accurate volume calculation exists, and was stored
                                            #if triang=false suceeded for some misterious reason, I use Arvo here. I don't expect this happens..
                                            if(err.value==1):
                                                print("ACCURATE:Error in volume computation of pocket "+str(pn)+" for parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max))
                                                print("Trying with analytic approach..")
                                                volume,area,err = p['node'].volume()
                                                print(volume,area,err)
                                    
                                            # print("p"+str(pn)+" accurate area and vol:"+str(volume)+","+str(area))

                                            resData = clustering.getRes(p['node'].getAtomsIds(),getCoord=True,ligandCoord=l['coord'])
                                            entrances_sizeBased = getEntrance(p['mouths'],pwDist=True)
                                            entrances_rBased = getEntrance(p['all_mouths'],pwDist=True)
                                            quality_correlator={"name":structures[s]['pqr'],"parameters":(alpha,beta,rp_max),"score":(OS,VS),"size":p['node'].count,
                                            "volume":volume,"area":area, "res":resData,"entrances_sizeBased":[e[1:] for e in entrances_sizeBased],
                                            "entrances_rBased":[e[1:] for e in entrances_rBased],
                                            "persistence":p['persistence'],"top":(p['mouths'][0][1],p['mouths'][0][0].r),"bottlenecks":p['btlnk_info'],
                                            "aggregations":[a[1:] for a in p['aggregations']],"n_subs":len(p['subpockets']),"isub":False,"isPeptide":l["isPeptide"]}
                                            pstream.storeDict(quality_correlator) #write persitence in python pickle format
                                            # if((OS>OS_max)and(gotHit)):
                                            if(OS>OS_max):
                                                #SAVE ONLY BEST MATCH PER LIGAND FOR STATISTICS
                                                hit =1
                                                # print("Previous score for current ligand (0=no previous hit in the pocket list -->with smaller score)",OS_max)
                                                OS_max=OS
                                                VS_kept =VS
                                                if(saveP):
                                                    if not os.path.exists(saveP_path):
                                                        os.makedirs(saveP_path)
                                                    saveResSimple(pn,saveP_path,p['node'],clustering.get_protein().resMap)
                                                #print("Got HIT!\t pocket number:%d, with ligand: %s. Scores: OV= %.3f\tVS=%.3f"%(pn,l['name'],OS,VS))
                                        OS_sub_max = 0
                                        for sn,sub in enumerate(p['subpockets']):
                                            sub['node'].load_protein(clustering.get_protein())
                                            gotHit,OS_sub,VS_sub = sub['node'].matchScore(l['coord'])
                                            # if(OS>0):
                                            if(gotHit):
                                                #More efficient: compute volume only if needed
                                                # print('succesfull subpocket')
                                                s_volume,s_area,err = sub['node'].NSVolume(accurate=True)
                                                # print("sub"+str(sn)+" accurate area and vol:"+str(s_volume)+","+str(s_area))
                                                if(err.value==1):
                                                    # if(ln==0):
                                                    #     err.info= "Error in volume computation of subpocket"+str(sn)+"in p"+str(pn)+" for parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max)+":  "+err.info + "\nTrying with ARVO module computing volume of overlapping spheres"
                                                    #     err.handle(errFile)
                                                        # print("Error in volume computation of subpocket"+str(sn)+" in p"+str(pn)+" for parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max))
                                                        # print("Trying with analytic approach..")
                                                    s_volume,s_area,err = sub['node'].volume()
                                                # print(volume,area)
                                                    if (err.value==1):
                                                        print("SKIPPING saving of structure= "+proteinFile+": Cannot compute volume of subpocket"+str(sn)+" in p"+str(pn)+" for parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max))
                                                        err.info="\n SKIPPING storage :Cannot compute volume of subpocket"+str(sn)+"in p"+ str(pn) +" for parameters: alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) + "of structure= "+proteinFile
                                                        err.handle(errFile)
                                                        continue
                                                    # else:
                                                    #     print("ok!")
                                                resData = clustering.getRes(sub['node'].getAtomsIds(),getCoord=True,ligandCoord=l['coord'])
                                                quality_correlator={"name":structures[s]['pqr'],"parameters":(alpha,beta,rp_max),"score":(OS_sub,VS_sub),"size":sub['node'].count,
                                                "volume":s_volume,"area":s_area,"res":resData,"entrances_sizeBased":[(sub['depth'],sub['rtop'],sub['rtop'],1)],"entrances_rBased":[(sub['depth'],sub['rtop'],sub['rtop'],1)],
                                                "persistence":sub['persistence'],"top":(sub['depth'],sub['rtop']),"bottlenecks":sub['btlnk_info'],
                                                "aggregations":[a[1:] for a in sub['aggregations']],"n_subs":0,"isub":True,"isPeptide":l["isPeptide"]}
                                                pstream.storeDict(quality_correlator) #write persitence in python pickle format
                                                # if((OS>OS_max)and(gotHit)):
                                                # Overwrite with score of subpockets when present
                                                #     hit =1
                                                #     OS_max=OS
                                                #     VS_kept =VS
                                                if((OS==OS_max) and (OS_sub>OS_sub_max) and saveP):
                                                    OS_sub_max = OS_sub
                                                    #Save sub pocket only of best pocket
                                                    if not os.path.exists(saveP_path):
                                                        os.makedirs(saveP_path)
                                                    saveResSimple(pn,saveP_path,sub['node'],clustering.get_protein().resMap,nsub=sn)
                                    volume_t=volume_t/pCounter              
                                except ContinueI:
                                    #Raise continueI to go here..
                                    
                                    print("\nPROBLEM: SKIPPING COMPARESON WITH CURRENT LIGAND")
                                    logFile.write("\n A MAJOR WARNING WAS PRODUCED\n")
                                    err.info = err.info + "\n Skipping current Ligand: "+ l['name']
                                    err.value = 1 #prevents exit, in this case we want to skip
                                    err.handle(errFile)
                                    continue
                                if(saveLocal):
                                        localHitFile[ln].write("\n%.2f\t%.2f\t%.2f\t%.3f\t\t%.3f\t\t\t%d"
                                        %(alpha,beta,rp_max,np.round(OS_max,3),np.round(VS_kept,3),pn+1))
                                    # print("\n LIGAND %s: %.1f\t%.1f\t%.1f\t%.3f\t%.3f"%(l['name'],alpha,beta,rp_max,OS_max,VS_kept))
                                    
                                    ########## new ########
                                if(hit==0):
                                    if((alpha,beta,rp_max)in nohitMap):
                                        nohitMap[alpha,beta,rp_max].append(structures[s]['pqr']+'_'+structures[s]['ligands'][ln])
                                    else:
                                        nohitMap[alpha,beta,rp_max] = [structures[s]['pqr']+'_'+structures[s]['ligands'][ln]]  
                                #####
                                #UPDATE COUNTERS
                                if (l["isPeptide"]==False):
                                    Hit_noP[i,j,k] +=hit #max 1 hit X structure-ligand
                                    OS_log_noP [i,j,k]+=OS_max
                                    VS_log_noP [i,j,k]+=VS_kept
                                    norm_noP [i,j,k] +=1 #doing so I correctly count only analysed structures (ligand+protein = 1 structure)   
                                    
                                else:
                                    pass

                                Hit[i,j,k] +=hit #max 1 hit X structure-ligand
                                OS_log[i,j,k]+=OS_max
                                VS_log[i,j,k]+=VS_kept
                                norm[i,j,k] +=1 #doing so I correctly count only analysed structures (ligand+protein = 1 structure)   

                                # print(norm)
                                                                        
                            # --------------- **Out from ligand loop** -------------------------

                            # THIS IS NOT LINKED TO LIGANDS ANALYSED BUT IS A GENERAL PROPERTY OF POCKETS FOUND 
                            # for each triplet of parameters x structure
                            nPockets[i,j,k] = len(pList)
                            nSubs[i,j,k] = nsubs
                            nSpheres[i,j,k] = nspheres
                            nSpheres_xP[i,j,k] = nspheres/len(pList) #n spheres x pocket
                            Volume[i,j,k] =volume_t
                        
                        except Exception:
                            print("\nProblem, skipping current triplet due to failure in pocket analysis\n")
                            logFile.write("\n A MAJOR WARNING WAS PRODUCED\n")
                            err.info = err.info + "\n Skipping current triplet due to failure in pocket analysis \n Skipping alpha= %.2f, beta= %.2f ,rp_max = %.2f," %(alpha,beta,rp_max) +"of current protein="+proteinFile
                            err.value = 1 #prevents exit, in this case we want to skip
                            err.handle(errFile)
                            #Skip current parameters triplet
                            continue

                        processed_structures[i,j,k]+=1



            #END OF 1st NESTED CYCLE: all parameters over given structure
            print("\n\n ** -- Done -- *** ")
            nAnalysed += len(ligands_coord) #n structure ligand pairs
            nAnalysed_noP = nAnalysed - nPeptides
           
            advancement =(100*np.round((s+1)/n_structures,3),s+1-skipped_structures,n_structures)
            print("ADVANCEMENT: %d %% of structures"%advancement[0])
            

            ########### current averages
            avHit = np.nan_to_num(Hit/norm)
            average_OS_log=np.nan_to_num(OS_log/Hit)#average only over success
            average_VS_log=np.nan_to_num(VS_log/Hit)
            ###
            sum_nPockets +=nPockets
            nPockets = nPockets/np.amin(nPockets)#number of pockets for each parameter divided by minimum number of pockets found in the structure for each param
            sum_norm_nPockets +=nPockets

            sum_nSubs +=nSubs #number of subpockets WITHOUT counting parent pocket (a simple pocket with no sub - pocket is not counted here)
            nSubs = np.nan_to_num(nSubs/np.amin(nSubs)) 
            sum_norm_nSubs +=nSubs

            sum_nSpheres += nSpheres #total probe spheres 
            nSpheres = nSpheres/np.amin(nSpheres) 
            sum_norm_nSpheres += nSpheres

            sum_nSpheres_xP += nSpheres_xP #probe spheres x pocket for all parameters over single structure
            nSpheres_xP =  np.nan_to_num(nSpheres_xP/np.amin(nSpheres_xP))
            sum_norm_nSpheres_xP += nSpheres_xP

            sum_Volume +=Volume
            Volume = np.nan_to_num(Volume/np.amin(Volume))
            norm_Volume +=Volume 

            avHit_noP = np.nan_to_num(Hit_noP/norm_noP)
            average_OS_log_noP=np.nan_to_num(OS_log_noP/Hit_noP)#average only over success
            average_VS_log_noP=np.nan_to_num(VS_log_noP/Hit_noP)
            ###


            # processed_structures = s+1-skipped_structures
            save_avHit(alphas,betas,radii,average_OS_log,average_VS_log,avHit,
            sum_nPockets/(processed_structures),sum_norm_nPockets/(processed_structures),sum_nSubs/(processed_structures),
            sum_norm_nSubs/(processed_structures),sum_Volume/processed_structures,norm_Volume/processed_structures,
            sum_nSpheres_xP/(processed_structures),sum_norm_nSpheres_xP/(processed_structures),sum_nSpheres/(processed_structures),sum_norm_nSpheres/(processed_structures),
            advancement,nAnalysed,date=Crono().init())

            #Cleaned from peptides:
            save_avHit(alphas,betas,radii,average_OS_log_noP,average_VS_log_noP,avHit_noP,
            sum_nPockets/(processed_structures),sum_norm_nPockets/(processed_structures),sum_nSubs/(processed_structures),
            sum_norm_nSubs/(processed_structures),sum_Volume/processed_structures,norm_Volume/processed_structures,
            sum_nSpheres_xP/(processed_structures),sum_norm_nSpheres_xP/(processed_structures),sum_nSpheres/(processed_structures),sum_norm_nSpheres/(processed_structures),
            advancement,nAnalysed_noP,date=Crono().init(),filename="Nopeptide_summary_hitFile.out")

            # FLUSH BUFFER 
            t=meso_crono.get()
            print(t)
            logFile.write(t+"\n")
            logFile.write("ADVANCEMENT: %d %% of structures"% advancement[0])
            logFile.flush()
            errFile.flush()
            if(saveLocal):
                for lfile in localHitFile:
                    lfile.close()

            
        ############## END OF MAIN CYCLE ######################
        # ######### write structures which failed for given parameters
        for alpha in alphas:
                for beta in betas:
                    for rp_max in radii:
                        if((alpha,beta,rp_max)in nohitMap):
                            ff.write("\n%.2f\t%.2f\t%.2f" %(alpha,beta,rp_max))
                            ff.write(repr(nohitMap[alpha,beta,rp_max]).replace("[","\n").replace("]","").replace(", ","\n"))
        ff.close()
        ##################

        # if(not lightMode):
        pstream.end()

        err.handle(errFile)

    else:
        #SINGLE RUN MODE
        print("\n*SINGLE RUN*\n")
        logFile.write("\n***SINGLE RUN MODE (USING PROVIDED PARAMETERS OR DEFAULT)***\n\n")
        logFile.write("Protein structure name= "+proteinFile+"\n")
        logFile.write("Config parameters are:\n alpha= %.1f beta= %.1f \n Maximum probe radius= %.1f, minimum probe radius= %.1f, probe increment= %.2f" 
        % (alpha,beta,rp_max,global_module.R_WATER,global_module.delta_rp))
        print("Config parameters are:\n alpha= %.1f beta= %.1f \n Maximum probe radius= %.1f, minimum probe radius= %.1f, probe increment= %.2f\n" 
        % (alpha,beta,rp_max,global_module.R_WATER,global_module.delta_rp))
        print("Protein structure name= "+proteinFile+"\n")

        clustering = NS_clustering()
        err = clustering.init(proteinFile)

        err.handle(errFile)
        
        err,pList=clustering.build(alpha,beta,rpMAX = rp_max)

        
        err.handle(errFile)
        logFile.write("\n\n Number of (large) pockets found: "+str(len(pList)))

        if(makeInfo):
            #also performs score and size filtering 
            if(pList):
                # err,pList=clustering.printInfo_old(saveSpheres = True,largeP=global_module.largeP)
                err,pList_ranked = clustering.printInfo(saveSpheres=True,type="IF10_noDensity",mode=1)
                logFile.write("\n Detailed info in protein output file("+global_module.pdbFolder_path+") and status.txt file("+global_module.runFolder_path)
                err.handle(errFile)
            else:
                print("NO pockets found with current parameters!")
        if(global_module.accTriang):
            clustering.VMD_accTriang()


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

    return 
##################
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser exit")
        sys.exit()
