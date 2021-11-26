# Sudy of new ranking schemes..
#
#
#

import sys
import os
import subprocess
import numpy as np

from pickPocket import *
from pickPocket import global_module


np.seterr( invalid='ignore') #avoids warning when no match causes 0/0 division when gathering stats


makeInfo = True
saveLocal = False
saveP = False  # save res and number of hitting pocket.. CAREFUL use for single clustering parameter!

excludePeptide = False
VS_threshold = 0.5
OV_threshold = 0.5

print("THRESHOLDS:\nOV=%.1f\tVS=%.1f"%(OV_threshold,VS_threshold))

amino = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','HID','HIE','HIP','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}

class ContinueI(Exception):
    pass

################## MAIN ###############################
def main():
    inlineCall = False
    # INITIALIZATIONS #
    runPath = 'temp'
    if not os.path.exists(runPath):
        os.makedirs(runPath)
        subprocess.call('cp '+global_module.pathTo_NS_ex+'* '+runPath+"/", shell=True)
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
        from pickPocket.train_classifier import Scoring,getFeatNoDensity
        from pickPocket.functions import save_rankingStats,hitCounter,saveALLhitStats,saveFineRankingStats,save_rankingStats_simple

        #   *********** LOAD TRAINED MODEL ****************

        
        print("\n++++TEST MODE: assessing performance of predictor on structures +++++\n")#generic, could be training as new samples
        logFile.write("\n++++TEST MODE: assessing performance of predictor on structures +++++\n")
        # +++++++++++++++++++ Load trained classifier +++++++++++++++++++++
        
        scoreIF = Scoring()
        
        modelName="IF10" 
        # model2="IF10_onlyDensity" # Hydro density replaces hydro count
      
        
        #names = ["IF_1.out","IF_2.out","IF_15.out","IF_20.out"]
        names = ["IF_1.out","IF_2.out","IF_3.out","IF_4.out"]

        err = scoreIF.load(modelName,modelType=1,unique=False) #change here and below to test different classifiers..
        # err = scoreIF_onlyDensity.load(model2,modelType=1,unique=False)

        
        err.handle(errFile)

        n_models = 4 # Ranking modes..
        penalty=[0.01,0.02,0.03,0.04]

        hitTop3 = np.zeros((alphas.size,betas.size,radii.size,n_models))
        hitTop10 = np.zeros((alphas.size,betas.size,radii.size,n_models))
        hitMatrix = np.zeros((alphas.size,betas.size,radii.size,n_models,10))
       

        allHit = np.zeros((alphas.size,betas.size,radii.size,n_models))
        OS_log = np.zeros((alphas.size,betas.size,radii.size,n_models))
        VS_log = np.zeros((alphas.size,betas.size,radii.size,n_models))
        volume_log = np.zeros((alphas.size,betas.size,radii.size,n_models))

        norm = np.zeros((alphas.size,betas.size,radii.size,n_models))
        singleHit = np.zeros((alphas.size,betas.size,radii.size,n_models))
        topSubHit = np.zeros((alphas.size,betas.size,radii.size,n_models))
        
        
        avScore = np.zeros((alphas.size,betas.size,radii.size,n_models))
        avScoreTop3 = np.zeros((alphas.size,betas.size,radii.size,n_models))
        # average_nSubs = np.zeros((alphas.size,betas.size,radii.size,n_models))

        nAnalysed = 0
        nPockets=np.empty((alphas.size,betas.size,radii.size))
        sum_nPockets=np.zeros((alphas.size,betas.size,radii.size))

        # nohitMap =[{}] * n_models
        nohitMap =[]
        for i in range(n_models):
            nohitMap.append({})
        
        ff = [open("noTopHit_1percent_"+modelName+".out",'w'),open("noTopHit_2percent_"+modelName+".out",'w'),open("noTopHit_3percent_"+modelName+".out",'w'),
        open("noTopHit_4percent_"+modelName+".out",'w')]


        #LOOP OVER STRUCTURES
        skipped_structures = 0
        clustering = NS_clustering()
        analStructures = 0 
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
                        
                        numberPockets = len(pList)

                        featListGeom,featListChem,map_direct,map_reverse = getFeatNoDensity(pList,clustering.get_protein().resMap)
                        
                        
                        scoreIF.resetRank()
                        rankedIndexesIF,rankedIndexesSubIF,numericalScoreIF,numericalScoreSubIF = scoreIF.getRanking_noSorting(featListGeom,featListChem)
                        rankIF_geomChem = (rankedIndexesIF,rankedIndexesSubIF)
                        score_geomChem = (numericalScoreIF,numericalScoreSubIF)
                        

                        ##### NEW RANK FLATTENING SUBS..
                        # Here no distinction, 
                        # 1. each pocket with 1 subpocket is substituted by it.
                        # 2. each pocket with more that 1 sub is relaced by its subs with dedicated ranking
                        #Strategy (most efficient): update on the fly rankedIndexes and respective map..



                        
                        ranks =rankIF_geomChem
                        scores = score_geomChem

                        save_path = 'output/'+structures[s]['pqr']+'_Pres'
                        for ln,l in enumerate(ligands_coord):
                            print(l['name'])
                            # top3 = np.zeros(n_models)
                            # top10 = np.zeros(n_models)
                            # topwithSub =np.zeros(n_models)
                            # topOnlySub = np.zeros(n_models)
                            # top1Sub = np.zeros(n_models)

                            OS_kept = np.zeros(n_models)
                            VS_kept = np.zeros(n_models)

                            #NEW
                            volume = np.zeros(n_models)
                            
                            for m in range(n_models):
                        

                                print("\nmodel penalty = ",penalty[m])
                                # save_path = 'output/'+str(m)+'/'+structures[s]['pqr']+'_Pres'

                                rankMain = ranks[0]
                                rankSub =  ranks[1]# For some ranking modes it coincides with the above..
                                scoreMain = scores[0]
                                scoreSub = scores[1]
                                #ligand-structure loop
                                r=0 # rank position
                                r_in=0 # running over ranked indexes array
                                selfContained = set()

                                # print('original rank')
                                # print(np.sort(scoreMain)[:10])
                                # print(rankMain[:10])

                                while ((r<10) and (r_in < rankMain.size)):

                                    

                                    pi,si = map_direct[rankMain[r_in]] #pocket and subpocket original index



                                    # print("rank: %d,  index: %d"%(r+1,rankMain[r_in]))
                                    if(si is not None):
                                        # print("I'm an original subpocket")
                                        # ------------------- SUBPOCKET  (TREATED AS MASTER FROM USER PERSPECTIVE)--------------------
                                        # if(pi in selfContained):
                                        #     #going to next element (positions do not progress in the ranking) since sub contained in master pocket higher in ranking
                                        #     r_in+=1
                                        #     continue # r index does not advance
                                        n_subs=0
                                        pocket = pList[pi]['subpockets'][si]['node'] #IS A SUBPOCKET
                                        pocket.load_protein(clustering.get_protein())
                                        # Question: evaluate goodnes of subpocket based only on volume score? (much more severe..)
                                        gotHit,OS,VS = pocket.matchScore(l['coord'],coverageTh=VS_threshold,matchScoreTh=OV_threshold)
                                        if gotHit:
                                            # SUBPOCKET HIT TREATED AS SINGLE POCKET
                                            print("singleP hit")
                                            hitMatrix[i,j,k,m,:] += hitCounter(r)

                                            
                                            allHit[i,j,k,m] +=1
                                            OS_kept[m] = OS
                                            VS_kept[m] = VS
                                            volume[m],_A,_err = pocket.volume()

                                            hitTop10[i,j,k,m] +=1
                                            score=scoreMain[rankMain[r_in]]
                                            avScore[i,j,k,m] += score
                                            
                                            # print('score = ', score)
                                            # print('unnormalized avScore = ',avScore[i,j,k,m], 'norm = ',hitTop10[i,j,k,m])

                                            singleHit[i,j,k,m] +=1 #counter of how many times pocket with 0 or 1 subpockets hit

                                            if(r<3):#within top3
                                                hitTop3[i,j,k,m] += 1
                                                avScoreTop3[i,j,k,m] +=score
                                            if(saveP and m==0):
                                                if not os.path.exists(save_path):
                                                    os.makedirs(save_path)
                                                saveResSimple(r,save_path,pocket,clustering.get_protein().resMap)
                                            break
                                                                                
                                        # removing redundancy on single subpocket. If single subpocket higher ranked, I consider it redundant with master one 
                                        
                                        if(len(pList[pi]['subpockets'])==1):
                                        #Single subpocket--> master pocket in black list
                                            selfContained.add(pi)

                                    else:
                                        gotHit = False #'master pocket hit'
                                        # --------- PARENT POCKET (or with no subpockets) --------------------
                                        if(pi in selfContained): #SKIP PARENT POCKET OF A SINGLE SUBPOCKET ALREADY EXPRESSED IN THE RANKING
                                            r_in+=1
                                            continue
                                        selfContained.add(pi) #to filter out subpockets already expressed by the master pocket. NOW USELESS
                                        subs = pList[pi]['subpockets']
                                        n_subs = len(subs)
                                        score=scoreMain[rankMain[r_in]]
                                        if(n_subs==0):
                                            pocket = pList[pi]['node']
                                            pocket.load_protein(clustering.get_protein())
                                            gotHit,OS,VS = pocket.matchScore(l['coord'],coverageTh=VS_threshold,matchScoreTh=OV_threshold)
                                            if gotHit:
                                                print("singleP hit")
                                                singleHit[i,j,k,m] += 1 #counter of how many times pocket with 0 or 1 subpockets hit
                                                allHit[i,j,k,m] +=1
                                                hitMatrix[i,j,k,m,:] += hitCounter(r)
                                                OS_kept[m] = OS
                                                VS_kept[m] = VS
                                                volume[m],_A,_err = pocket.volume()
                                                avScore[i,j,k,m] += score
                                                hitTop10[i,j,k,m] +=1
                                                if(r<3):
                                                    hitTop3[i,j,k,m] += 1
                                                    avScoreTop3[i,j,k,m] += score
                                                break #if no subpockets ok I can break here
                                            r+=1
                                            r_in+=1 
                                            continue
                                                                               
                                        internal_rank=[] # container for score of subpockets analysed 
                                        
                                        indS = []
                                        
                                        for sub_i in range(n_subs):
                                            try:
                                                internal_rank.append(np.where(rankSub==map_reverse[(pi,sub_i)])[0][0]) #list with subranks of inquired pockets
                                            except KeyError:
                                                #The subpocket has been skipped for some reasons (Nan in feature vector or cannot compute V..)
                                                print("Tried to find skipped subpocket")
                                                internal_rank.append(rankSub.size) # end of ranking
                                                scoreSub = np.append(scoreSub,1)
                                                map_reverse[(pi,sub_i)] = rankMain.size 
                                                numericalScoreIF = np.append(numericalScoreIF,1)
                                                map_direct[1] = (pi,sub_i)
                                        indS = np.argsort(internal_rank) #relative rank among subpockets
                                        # print(indS)
                                        for ns,sub_i in enumerate(indS):
                                            # print('sub:', ns +1, 'index:',map_reverse[(pi,sub_i)] )
                                            if(ns==0):
                                                pocket = subs[sub_i]['node']
                                                pocket.load_protein(clustering.get_protein())
                                                gotHit,OS,VS = pocket.matchScore(l['coord'],coverageTh=VS_threshold,matchScoreTh=OV_threshold)
                                                if gotHit:
                                                    print("topSub hit")
                                                    if(n_subs==1):
                                                        # print('it was a single subpoket hit..')
                                                        singleHit[i,j,k,m] += 1 #counter of how many times pocket with 0 or 1 subpockets hit
                                                    allHit[i,j,k,m] += 1
                                                    hitMatrix[i,j,k,m,:] += hitCounter(r)
                                                    OS_kept[m] = OS
                                                    VS_kept[m] = VS
                                                    volume[m],_A,_err = pocket.volume()
                                                    avScore[i,j,k,m] += score
                                                    hitTop10[i,j,k,m] +=1
                                                    topSubHit[i,j,k,m] +=1
                                                    if(r<3):
                                                        hitTop3[i,j,k,m] += 1
                                                        avScoreTop3[i,j,k,m] += score
                                                    break 
                                            else:
                                                scoreMain[map_reverse[(pi,sub_i)]] = score*(1.+penalty[m]*ns)
                                                # print("updating ranks")
                                                # print("index: ", map_reverse[(pi,sub_i)])
                                                # print("master score=%.3f-->%.3f"%(score,score*(1.+penalty[m]*ns)))
                                                rankMain = np.argsort(scoreMain) #update rank with new value for subpocket
                                                # if(ns==(n_subs-1)):
                                                #     print(np.sort(scoreMain)[:10])
                                                #     print(rankMain[:10])   
                                        if gotHit:
                                            print("subpocket %d hit"%(ns +1))
                                            # print("rank=",r+1)
                                            break
                                        

                                    r_in+=1 #ACTUAL INDEX FOR LOOPING THROUGH POCKETS.. while r is the appearing rank
                                    r+=1
                                if((r==10) or  (r_in == rankMain.size)):                                   
                                    if((alpha,beta,rp_max)in nohitMap[m]):
                                        nohitMap[m][alpha,beta,rp_max].append(structures[s]['pqr']+'_'+structures[s]['ligands'][ln])
                                    else:
                                        nohitMap[m][alpha,beta,rp_max]=[structures[s]['pqr']+'_'+structures[s]['ligands'][ln]]

                            OS_log[i,j,k,:]+=OS_kept
                            VS_log[i,j,k,:]+=VS_kept
                            volume_log [i,j,k,:]+=volume
                            norm[i,j,k,:] +=1
                           

                        #out of ligand loop    
                        nPockets[i,j,k] = numberPockets
                        
                        
            #OUT OF ALL PARAMETER LOOPS
            analStructures += 1.
            nAnalysed += len(ligands_coord) #n structure ligand pairs 
            sum_nPockets+=nPockets               
            avHitTop3 = np.nan_to_num(hitTop3/norm)
            avHitTop10 = np.nan_to_num(hitTop10/norm)
            #normForsub = hitTop10-singleHit #count when subs are succesfull among all hits which are not "single" hits
            #norm - singleHit #corrNorm: when master with no sub or sigle subpocket hit

            

            avSingleHit = singleHit/norm
            avTopSubHit = topSubHit/norm
            

            average_OS_log=np.nan_to_num(OS_log/allHit)#average only over success
            average_VS_log=np.nan_to_num(VS_log/allHit)
            average_volume_log = np.nan_to_num(volume_log/allHit)
            

            #Averaged scores

            avScore_log=np.nan_to_num(avScore/hitTop10)
            avScoreTop3_log = np.nan_to_num(avScoreTop3/hitTop3)
            
            # SAVING METHOD.. 
            for m in range(n_models):
                save_rankingStats_simple(m,alphas,betas,radii,avHitTop3,avHitTop10,avTopSubHit,
                avSingleHit,average_OS_log,average_VS_log,average_volume_log,sum_nPockets/analStructures,
                avScore_log,avScoreTop3_log,nAnalysed,names,date=Crono().init())
                saveALLhitStats(m,alphas,betas,radii,hitMatrix,allHit,nAnalysed,names,date=Crono().init())


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
    ####################### STATS ON MANY PARAMETERS
    elif (isAnalysis): 
        from pickPocket.functions import getEntrance,writeBinary,save_avHit
    
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
