
import sys
import os
import subprocess
import numpy as np

from pickPocket import *
from pickPocket import global_module

np.seterr( invalid='ignore') #avoids warning when no match causes 0/0 division when gathering stats


makeInfo = True
saveLocal = False


saveP = True  # save res and number of hitting pocket..
testAlpha = 0.0
testBeta = 0.7



class ContinueI(Exception):
    pass

################## MAIN ###############################
def main():
    inlineCall = False
    # INITIALIZATIONS #
    runPath = 'temp'
    if not os.path.exists(runPath):
        os.makedirs(runPath)
# print(global_module.pathTo_NS_ex+'/*')
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
        from pickPocket.functions import save_rankingStats,hitCounter,saveALLhitStats,saveFineRankingStats,getRanking_volume,saveResSimple

        alpha = testAlpha
        beta = testBeta

        #   *********** LOAD TRAINED MODEL ****************

        print("\n++++TEST MODE: assessing performance of predictor on structures +++++\n")#generic, could be training as new samples
        logFile.write("\n++++TEST MODE: assessing performance of predictor on structures +++++\n")
        # +++++++++++++++++++ Load trained classifier +++++++++++++++++++++
        
        scoreIF = Scoring()
        
        # scoreEIF = Scoring()
        model1="IF10" #with new def of entrances based on radius and the 'buried' feature and chemical forest
       

        err = scoreIF.load(model1,modelType=1,unique=False) #change here and below to test different classifiers..
        
        
        err.handle(errFile)

        n_models = 1 # Ranking modes..


        nAnalysed = 0


        #LOOP OVER STRUCTURES
        skipped_structures = 0
        clustering = NS_clustering()
        for s in range(n_structures):
            print("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n" )
            logFile.write("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n")
            proteinFile = structures[s]['pqr']
            ligands = structures[s]['ligands']

            
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

    
            meso_crono.init()
            
        
            try:
                err,pList=clustering.build(alpha,beta,rpMAX = 3.0)
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
        
            # ************ RANKING *********************************8
            
            featListGeom,featListChem,map_direct,map_reverse = getFeatNoDensity(pList,clustering.get_protein().resMap)
            
            # featList = [fg + fc for fg,fc in zip(featListGeom,featListChem)]
            scoreIF.resetRank()
            rankedIndexesIF,rankedIndexesSubIF,_numericalScoreIF,_numericalScoreSubIF = scoreIF.getRanking(featListGeom,featListChem)

            rankIF = (rankedIndexesIF,rankedIndexesSubIF)

            
            save_path = 'output/'+structures[s]['pqr']+'_pockets'

            rankMain = rankIF[0]
            rankSub =  rankIF[1]# For some ranking modes it coincides with the above..
            #ligand-structure loop
            r=0 # rank position
            r_in=0 # running over ranked indexes array
            selfContained = set()

            # print(rankMain)
            # print()
            # print(map_direct)

            while ((r<10) and (r_in < rankMain.size)):
                pi,si = map_direct[rankMain[r_in]] #pocket and subpocket original index
                # print(r,r_in)
                if(si is not None):
                    # ------------------- SUBPOCKET --------------------
                    if(pi in selfContained):
                        #going to next element (positions do not progress in the ranking) since sub contained in master pocket higher in ranking
                        r_in+=1
                        continue # r index does not advance

                    pocket = pList[pi]['subpockets'][si]['node'] #IS A SUBPOCKET
                    
                    
                    if(len(pList[pi]['subpockets'])==1):
                    #Single subpocket--> master pocket in black list
                        selfContained.add(pi)

                    if(saveP):
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        saveResSimple(r,save_path,pocket,clustering.get_protein().resMap)

                else:
                    # --------- PARENT POCKET (or with no subpockets) --------------------
                    if(pi in selfContained): #SKIP PARENT POCKET OF A SINGLE SUBPOCKET ALREADY EXPRESSED IN THE RANKING
                        r_in+=1
                        continue
                    selfContained.add(pi) #to filter out subpockets already expressed by the master pocket

                    pocket = pList[pi]['node']
                   
                    subs = pList[pi]['subpockets']
                    n_subs = len(subs)
                    
                    if(saveP):
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        saveResSimple(r,save_path,pocket,clustering.get_protein().resMap)
                    
                    internal_rank=[] # container for score of subpockets analysed 
                    
                    indS = []# just to avoid an extra "if" clause
                    if(n_subs>1):
                        # Save subpockets only when more than one

                        for sub_i in range(n_subs):
                            internal_rank.append(np.where(rankSub==map_reverse[(pi,sub_i)])[0][0])
                            indS = np.argsort(internal_rank) #relative rank among subpockets
                        for ns,sub_i in enumerate(indS):
                            subpocket = subs[sub_i]['node']
                            subpocket.load_protein(clustering.get_protein())
                            if(saveP):
                                if not os.path.exists(save_path+'/sub'):
                                    os.makedirs(save_path+'/sub')
                                saveResSimple(r,save_path+'/sub',subpocket,clustering.get_protein().resMap,nsub=ns)
                r+=1
                r_in+=1
                
            advancement =(100*np.round((s+1)/n_structures,3),s+1-skipped_structures,n_structures)
            print("ADVANCEMENT: %d %% of structures"%advancement[0])
            t=meso_crono.get()
            print('\n',t)
            logFile.write(t+"\n")
            logFile.write("ADVANCEMENT: %d %% of structures"% advancement[0])
            logFile.flush()
            errFile.flush()
        #Summary of structures-Ligand not classified into top 10
        
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
        for s in range(n_structures):
            #MAIN LOOP OVER STRUCTURES
            print("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n" )
            logFile.write("\n\n ******** Current analysis: " + structures[s]['pqr'] + " ********** \n")
            proteinFile = structures[s]['pqr']
            ligands = structures[s]['ligands']

            # print("Loading ligand(s) coordinates")
            ligands_coord=[]
            localHitFile=[]

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

            for ligand_name in ligands:
                # print(ligand_name)    
                try:
                    coord = readLigands(global_module.pdbFolder_path+ligand_name,proteinCoord=clustering.get_protein().atoms)
                    if (len(coord)==0):
                        print("Skipping "+ligand_name+" of structure "+proteinFile+ " : No ligands heavy atoms within 5A from protein")
                        err.value = 1
                        err.info = "Skipping "+ligand_name+" of structure "+proteinFile + " : No ligands heavy atoms within 5A from protein!"
                        err.handle(errFile)

                        continue
                    ligands_coord.append({"name":ligand_name,"coord":coord})
                    if(saveLocal):
                        fp = open(global_module.pdbFolder_path+"hitfile_"+proteinFile+"_"+ligand_name+".out",'w')
                        fp.write("# ** Hit file for "+ proteinFile+" with ligand "+ligand_name+ "**")
                        fp.write('\n#'+ Crono().init())
                        fp.write("\n# alpha\tbeta\trp_max\tOverlapScore\tVolumeOverlap\tNumber of pockets found")
                        localHitFile.append(fp)
                except NameError:
                    err.value = 1
                    err.info = "Ligand " + ligand_name +" could not be loaded. \n Skipping, it won't be considered among available comparesons. "
                    err.handle(errFile)
                    print("SKIPPING: Ligand  "+ligand_name+ " could not be loaded")
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
                                            quality_correlator={"parameters":(alpha,beta,rp_max),"score":(OS,VS),"size":p['node'].count,
                                            "volume":volume,"area":area, "res":resData,"entrances_sizeBased":[e[1:] for e in entrances_sizeBased],
                                            "entrances_rBased":[e[1:] for e in entrances_rBased],
                                            "persistence":p['persistence'],"top":(p['mouths'][0][1],p['mouths'][0][0].r),"bottlenecks":p['btlnk_info'],
                                            "aggregations":[a[1:] for a in p['aggregations']],"n_subs":len(p['subpockets']),"isub":False}
                                            pstream.storeDict(quality_correlator) #write persitence in python pickle format
                                            # if((OS>OS_max)and(gotHit)):
                                            if(OS>OS_max):
                                                #SAVE BEST MATCH PER LIGAND FOR STATISTICS
                                                hit =1
                                                # print("Previous score for current ligand (0=no previous hit in the pocket list -->with smaller score)",OS_max)
                                                OS_max=OS
                                                VS_kept =VS
                                                #print("Got HIT!\t pocket number:%d, with ligand: %s. Scores: OV= %.3f\tVS=%.3f"%(pn,l['name'],OS,VS))
                                        for sn,sub in enumerate(p['subpockets']):
                                            sub['node'].load_protein(clustering.get_protein())
                                            gotHit,OS,VS = sub['node'].matchScore(l['coord'])
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
                                                quality_correlator={"parameters":(alpha,beta,rp_max),"score":(OS,VS),"size":sub['node'].count,
                                                "volume":s_volume,"area":s_area,"res":resData,"entrances_sizeBased":[(sub['depth'],sub['rtop'],sub['rtop'],1)],"entrances_rBased":[(sub['depth'],sub['rtop'],sub['rtop'],1)],
                                                "persistence":sub['persistence'],"top":(sub['depth'],sub['rtop']),"bottlenecks":sub['btlnk_info'],
                                                "aggregations":[a[1:] for a in sub['aggregations']],"n_subs":0,"isub":True}
                                                pstream.storeDict(quality_correlator) #write persitence in python pickle format
                                                # if((OS>OS_max)and(gotHit)):
                                                # Overwrite with score of subpockets when present
                                                #     hit =1
                                                #     OS_max=OS
                                                #     VS_kept =VS
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


            # processed_structures = s+1-skipped_structures
            save_avHit(alphas,betas,radii,average_OS_log,average_VS_log,avHit,
            sum_nPockets/(processed_structures),sum_norm_nPockets/(processed_structures),sum_nSubs/(processed_structures),
            sum_norm_nSubs/(processed_structures),sum_Volume/processed_structures,norm_Volume/processed_structures,
            sum_nSpheres_xP/(processed_structures),sum_norm_nSpheres_xP/(processed_structures),sum_nSpheres/(processed_structures),sum_norm_nSpheres/(processed_structures),
            advancement,nAnalysed,date=Crono().init())

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
                err,pList_ranked = clustering.printInfo(saveSpheres=True,type="IF10",mode=1)
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
