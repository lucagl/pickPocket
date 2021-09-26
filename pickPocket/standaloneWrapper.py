################### 
# MAIN MODULE OF CLUSTERING ALGORITHM


############ Clustering of SES probes for pocket detection and  reconstruction ############
#Author:								Luca Gagliardi 
#Copiright:					    © 2020 Istituto Italiano di Tecnologia   
############################################################### 

#IMPORTANT NOTE:
# Order matters and a precise choice has been made:
# Clustering order depends on distance sorting. In this sense is hierarchical.
# Current questions
#   Tuning of shift is crucial.
# Bottleneck mouths filtered so that only smallest radius kept if multiple probes for same btlnk
# Con_shiftList (persistence/mouths) filetered keeping largest probe and depth information for redundancy

        ### GATHERING PARAMETERS

        #   o   o   o   o  <--> alpha            
        #     o   o   o     ↕ beta

 #tag : tag of the node of last aggregation
 # -1 singleton or aggregation with singleton
 # aggregation between non-singleton clusters:
#   0 lateral
#   4 pyramidal
 # 2 bottleneck
 # 3 conical shift (only if at least one of the two clusters is a singleton)
###################################################################################################


from pickPocket import global_module
from pickPocket.global_module import Crono
from pickPocket.global_module import Error

from pickPocket.data_containers import ClusterNode
from pickPocket.protein_class import Protein

from pickPocket.functions import new_probe,setup_NSInput,setUP_accurateTriang
from pickPocket.functions import Pdist_C,getIndex,crange

from pickPocket.train_classifier import getRank
from pickPocket.functions import saveP,saveRes

import numpy as np
import os


######### CLUSTERING PARAMETERS
shift_Threshold = global_module.shift_Threshold
count_Threshold = global_module.count_Threshold 
weight = count_Threshold
rmin = global_module.rmin
rmin_entrance = global_module.rmin_entrance

sin15_2 = global_module.sin15_2
R_WATER = global_module.R_WATER
tollerance = global_module.tollerance
###
############
conf = global_module.conf




class NS_clustering(object):

    def __init__(self):
        self.isInit = False
        self.pFilename = None
        self.save_path =None
        self.read_path = None
        self.log_path = None
        self.outFile = None

        self.protein = None
        self.ra = None
        #Careful, setting of global variables must have happened before object initialization
        self.pdbFolder_path = global_module.pdbFolder_path
        self.workingDir = global_module.runFolder_path
        self.filteredP = []
        return

    def init(self,structure_name):
        """
        Set file locations and correct folder paths.
        IMPORTANT: produces Nanoshaper_input.xyzr file
        """
        # #Update folder locations
        # self.pdbFolder_path = global_module.pdbFolder_path
        # self.runFolder_path = global_module.runFolder_path
        # self.workingDir = global_module.runFolder_path
        # ###
        err = Error()
        self.pFilename = structure_name
        self.save_path = self.pdbFolder_path+structure_name+"_Pfiles"
        self.read_path = self.pdbFolder_path+structure_name+'.pqr'
        self.log_path = self.workingDir+"status.txt"
        
        ### Container of structure atom positions and functions involving these atoms
        self.protein = Protein()
        try:
            self.protein.load(self.read_path)
        except Exception:
            err.value = 2
            err.info = "Error while loading pqr file.."
            return err
        self.ra = self.protein.atoms[:,3]
        self.outFile = open(self.log_path,'w')
        self.outFile.write("** TEMPORARY CURRENT RUN STATUS FILE **\n\n")
        self.outFile.write(" Protein analysed is: "+self.pFilename +"\n\n")

        ### Delicate.. if parallel I should initialise different working dirrectory for each protein
        # or have displace this in build() and add a atom clause here to avoid writing in the same time and void launching NS at the same time 
        np.savetxt(self.workingDir+"NanoShaper_input.xyzr",self.protein.atoms,delimiter="\t",fmt='%.4f')
        # print("Built Nanoshaper xyzr input file \n")
        self.outFile.write("\nBuilt Nanoshaper xyzr input file \n")
        
        self.outFile.write("\nSet up surfaceConfiguration.prm file \n")

        self.isInit=True

        return err
    def get_protein(self):
        return self.protein

    def VMD_accTriang(self):
        """
        Bring back NS surfaceConf to standard values.
        If pdbName given (single run mode), prepares an accurate triangulation of the structure for VMD.
        """
        print("\n ** Building accurate triangulation of structure..")
        new_probe(self.workingDir+conf,1.4)
        
        import subprocess
        setUP_accurateTriang(self.workingDir+conf)
        np.savetxt(self.workingDir+"NanoShaper_input.xyzr",self.protein.atoms,delimiter="\t",fmt='%.4f')
        try:
            #call NS
            out = subprocess.check_output('./NanoShaper',cwd=self.workingDir)
            # print(out)
        except subprocess.CalledProcessError as grepexc:                                                                                                   
            print ("error code", grepexc.returncode, grepexc.output)
            # print (out)
            # err.value = 2
            # err.info = "Errors in system call to NanoShaper.  "+str(grepexc.returncode) + str( grepexc.output)
            return 1
        subprocess.run(['mv',self.workingDir+"triangulatedSurf.face",self.pdbFolder_path+self.pFilename+'.face'])
        subprocess.run(['mv',self.workingDir+"triangulatedSurf.vert",self.pdbFolder_path+self.pFilename+'.vert'])
        
        return 0

    # def pocketMode(self,grid_scale,grid_selfInt,maxProbes_selfInt,gridPerfil=90,accTriang = False):
    #     setup_NSInput(self.workingDir+conf,grid_scale=grid_scale,grid_selfInt=grid_selfInt,maxProbes_selfInt=maxProbes_selfInt,accTriang=accTriang,pInput= True,gridPerfil=gridPerfil)
    #     new_probe(self.workingDir+conf,1.4)#use rp=1.4 to build pocket triangulation
        
    #     return


#####################
############

    def build(self,alpha,beta,rpMAX,isLight=False):
        import subprocess
        """
        Returns large pocket list with associated persistence vector and subpocket vector
        Produces a outfile (temporary, is rewritten upon successive calls)

        NOTE: Element count on large pockets is distint from count threshold since I want to allow for smaller subpockets.
        Only count threshold is a parameter defining the pocket. Large pockets are just a size filtered sublist of pockets.
        """
        setup_NSInput(self.workingDir+conf,accTriang=False)
        ClusterNode.not_firstTime = False
        # np.savetxt(self.workingDir+"NanoShaper_input.xyzr",self.protein.atoms,delimiter="\t",fmt='%.4f') # not clean, but needed if previously changed file for pocket analysis..

        err =Error()
        pcluster =[] #provisional container of pockets (need to be filtered)

        if(not self.isInit):
            err.value =2
            err.info = "The structure"+self.pFilename+" has not been initialized!"
            return err


       
        crono = Crono()
        self.outFile.write("\n"+crono.init()+"\n")
        
        
        # ra list of atomic radii of the protein
        
        # sin15_2 =   math.sin(math.pi/24.) #7.5°
        # largeThreshold = int(largeP * count_Threshold) #Used for further reduction of pockets returned by the function

        dendogram = {} #dictionary containing all clusters (a singleton cluster is a center) and associated distances
        #               [id-->clusterObject]
        aggregation_list = {} #Keeps trace of probes which have been clustered


        ################

        delta_rp = global_module.delta_rp
        probe,delta_rpNEW = crange(R_WATER,rpMAX,delta_rp,returnDelta=True)
        if(delta_rp!=global_module.delta_rp):    
            self.outFile.write("\n**Boundaries imposed a change to default delta_rp. Current delta_rp=%.3f \n" %np.round(delta_rp,3))
            delta_rp = delta_rpNEW

        nRun = len(probe)  
        # print(probe)
        self.outFile.write(str(probe))
        # print("# NS calls = ", nRun)
        self.outFile.write("\n number NS calls = " +str(nRun))
        # print("ALPHA: ", alpha)
        self.outFile.write("\n ALPHA: " +str(alpha))
        # print("BETA: ", beta)

        self.outFile.write("\n BETA: " +str(beta))
        
        ########### Useful Boundaries ###############
        
        ra_max = np.amax(self.ra)
        # ra_min=np.amin(ra) not used

        # print("Tollerance level bottlenecks: ", functions.tollerance*100, "%")
        self.outFile.write("\nTollerance level bottlenecks: %.1f%%\n\n" %(global_module.tollerance*100))
        #################

        ###Run

        n_run = 0
        n_total= 0
        id_shift = 0 
        global_shift = 0
        global_shift_old = 0
        # m is the current number of centers 
        scluster = []
        rp_old=0#dummy value
        coord_old = None
        for rp in probe:
            
            if(n_run>0):
                coord_old = coord
                ids_old = ids
            

            # print (n_run + 1, "********* CALL to NanoShaper **********\n")
            self.outFile.write( "\n********* CALL to NanoShaper " +str(n_run + 1)+" **********\n")
            # print("Probe radius = ", rp, "Previous radius =", rp_old)
            self.outFile.write("Probe radius = %.1f. Previous radius = %.1f" %(rp,rp_old))
            

            new_probe(self.workingDir+conf,rp)
            try:
                #call NS
                out = subprocess.check_output('./NanoShaper',cwd=self.workingDir)
            except subprocess.CalledProcessError as grepexc:                                                                                                   
                print ("error code", grepexc.returncode, grepexc.output)
                print (out)
                # print("\n\nHERE",flush=True)

                err.value = 2
                err.info = "Errors in system call to NanoShaper.  "+str(grepexc.returncode) + str( grepexc.output)
                return err,[]
            except Exception as e:
                err.info = "Unknown error in system call to NS.. " + str(e)
                err.value = 2
                return err,[]
            # print("-- DONE --")
            self.outFile.write("\n-- DONE --\n")

            data = np.loadtxt(self.workingDir+"centers.txt")

            # # Keep only not self intersecting
            # data = np.asarray(list(filter(lambda x: x[7] ==0,data)))

            try:
                coord = data[:,0:3] # current coordinates
                m = data.shape[0]
                ids = data[:,4:7].astype(int)
            except IndexError:
                err.value=2
                err.info = "Likely probe size too small for the current protein. "
                return err,[]
            except Exception as info:
                err.value=2
                err.info ="Unexpected exception: "+str(type(info))+ " while reading data produced by NanoShaper. "
                return err,[]
            ###
            n_total += m

            # print ("number regular probes: ",m) 
            self.outFile.write("number regular probes: " +str(m))

        #************* SINGLETON CLUSTERS ***************
        # Each probe is a singleton
            for k in range(id_shift,m+id_shift):
                dendogram[k]=[ClusterNode(k,coord=coord[k-id_shift],t_atoms=ids[k-id_shift],radius =rp),0]

            id_shift = len(dendogram)
        #***********************************

        # ********* SAME PROBE DISTANCES ************
            d_same,flag =  Pdist_C(coord,coord)

        # ********* BOTTLENECK TAG **************

        # --------------- PREFILTERING STEP ----------------
            sup_btlnk = np.sqrt(rp**2 +2*ra_max*rp)
            inf_btlnk = rp

            index = np.where((d_same<sup_btlnk)&(d_same>inf_btlnk))[0]
        #sort them according to distance
            mask=np.argsort(d_same[index])
            index = index[mask] 

            s = 0 #counter of aggregations at the current step

            for k in index:
                i,j=getIndex(flag,k,m)  
                answ,refIDS,refProbe = self.protein.is_btlnk(d_same[k],coord[i],ids[i],coord[j],ids[j],tollerance,indulgent=False)
                i = i+ global_shift
                j = j+global_shift

                if(answ):
                    myleft = i 
                    myright = j 
                    if(i in aggregation_list):
                        myleft = aggregation_list[i]
                    if(j in aggregation_list):
                        myright=aggregation_list[j]
                    if(myleft == myright):
                        #Do nothing they are already in same cluster
                        continue
                

                    #NEW NODE CREATED
                    dendogram[id_shift+s] = [ClusterNode(id_shift+s,refProbe,refIDS,rp,left =dendogram[myleft][0] , right =dendogram[myright][0] ,tag=2),d_same[k]]
                
                ########### Update aggregation list with topmost node label

                    for label in dendogram[id_shift+s][0].pre_order():
                        aggregation_list[label] = id_shift+s

                    #CHECK FOR POCKETS (preorder also updated self.tag)
                    if((dendogram[id_shift+s][0].tag == 1)and(dendogram[id_shift+s][0].count>=count_Threshold)and (dendogram[id_shift+s][0].r>=rmin)):
                        pcluster.append([dendogram[id_shift+s][0],dendogram[id_shift+s][0].index_btlnk,dendogram[id_shift+s][0].index_Cshift])  
                    ####
                    
                    s+=1

            id_shift = len(dendogram)
        #***********************

        # ********* PLAIN AGGREGATION ("shift") **************
            sup_shift =  min(sin15_2*(ra_max + rp),rp) *(1.+alpha)#min(0.1*(ra_max + rp),rp) *(1.+alphaC) #CRITICAL!!!!!!!!!

            index = np.where(d_same<sup_shift)[0]
            mask=np.argsort(d_same[index])
            index = index[mask] 

            s = 0
            
            for k in index:
                i,j=getIndex(flag,k,m) #original observations labels must be shifted accordingly
                
                answ = self.protein.is_shift(coord[i],ids[i],coord[j],ids[j])
                i = i +global_shift
                j = j+global_shift
                if(answ or (d_same[k]<delta_rp)):
                    # print(i,j)
                    # print(d_same[k])
                    myleft = i
                    myright = j 
                    tag = 0 
                # print(myleft,myright)
                    if(i in aggregation_list):
                        myleft = aggregation_list[i]
                    else:
                        tag = -1
                    if(j in aggregation_list):
                        myright=aggregation_list[j]
                    else:
                        tag = -1 
                    if(myleft == myright):
                        #Do nothing they are already in same cluster
                        continue
                    
                    #NEW NODE CREATED
                    dendogram[id_shift+s] = [ClusterNode(id_shift+s,None,None, radius = rp,left =dendogram[myleft][0] , right =dendogram[myright][0],tag=tag),d_same[k]]
                
                    for label in dendogram[id_shift+s][0].pre_order():
                        aggregation_list[label] = id_shift+s

                    #CHECK FOR POCKETS (preorder also updated self.tag)
                    if((dendogram[id_shift+s][0].tag == 1)and(dendogram[id_shift+s][0].count>=count_Threshold)and (dendogram[id_shift+s][0].r>=rmin)):
                        pcluster.append([dendogram[id_shift+s][0],dendogram[id_shift+s][0].index_btlnk,dendogram[id_shift+s][0].index_Cshift])  
                    ####

                    s+=1

            id_shift = len(dendogram) 
            #******************

            # ********* DIFFERENT PROBES DISTANCES ************

            if(n_run>0):
                #rp current run radius --> Largest
                d_diff,flag = Pdist_C(coord_old,coord) #IMPORTANT: keep this order in the arguments 


                #********* BOTTLENECKS AMONG DIFFERENT RADII ************

                #Note here reference probe has a gemotrical meaning. Its the probe related to the reference plane to the bottlenecj

                sup_btlnk = np.sqrt(rp**2 +2*ra_max*rp)
                inf_btlnk = rp

                index = np.where((d_diff<sup_btlnk)&(d_diff>inf_btlnk))[0]
            
                mask=np.argsort(d_diff[index])
                index = index[mask] 
        
                s = 0

                for k in index:
                    #i is old j is current
                    i,j=getIndex(flag,k,m) #original observations labels must be shifted accordingly
                
                    answ,refIDS,refProbe = self.protein.is_btlnk(d_diff[k],coord[j],ids[j],coord_old[i],ids_old[i],tollerance,indulgent=False)

                    if (np.array_equal(refProbe,coord[j])): 
                        ref_rad = rp
                    else: 
                        ref_rad = rp_old
                    
                    i = i +global_shift_old
                    j = j+global_shift

                    if(answ):
                        myleft = i
                        myright = j 

                        if(i in aggregation_list):
                            myleft = aggregation_list[i]
                        if(j in aggregation_list):
                            myright=aggregation_list[j]
                        if(myleft == myright):
                            #Do nothing they are already in same cluster
                            continue
                        
                        refIDS = ids[j-global_shift] #Take as current run
                        dendogram[id_shift+s] = [ClusterNode(id_shift+s,refProbe,refIDS,ref_rad,left =dendogram[myleft][0] , right =dendogram[myright][0] ,tag=2),d_diff[k]]
                        
                        for label in dendogram[id_shift+s][0].pre_order():
                            aggregation_list[label] = id_shift+s

                        #CHECK FOR POCKETS (preorder also updated self.tag)
                        if((dendogram[id_shift+s][0].tag == 1)and(dendogram[id_shift+s][0].count>=count_Threshold)and (dendogram[id_shift+s][0].r>=rmin)):
                            pcluster.append([dendogram[id_shift+s][0],dendogram[id_shift+s][0].index_btlnk,dendogram[id_shift+s][0].index_Cshift])  
                        ####
                        
                        s+=1

                id_shift=len(dendogram)

                # **************************

                # ***********  STRONG AND WEAK SHIFTS AGGREGATIONS ***********

                sup_shift = max(delta_rp,sin15_2*(ra_max+rp))*(1.+beta)

                index = np.where(d_diff<sup_shift)[0]
                mask=np.argsort(d_diff[index])
                index = index[mask] 

                
                s = 0
                for k in index:
                    #i is old j is current
                    i,j=getIndex(flag,k,m) #original observations labels must be shifted accordingly
                        
                    answ,local_tag = self.protein.is_shift(coord[j],ids[j],coord_old[i],ids_old[i],getTag=True)
                    #local tag= 4 : pyramidal connection

                    i = i +global_shift_old
                    j = j+global_shift

                # NEW :Tag 3 is given when aggregation happens with isolated NEW probe or for piramidal case. This should prevent some leackage..
                #t=0  o t=3 if 3 happens after 0 i might get taglist with successive 3s bu I would prefer not mixing the 2 clusters separated my a barrier..
                #o ||| o
                    if(answ):
                        # print(i,j)
                        # print(d_same[k])
                        myleft = i
                        myright = j #current run coordinate (LARGER PROBE)
                    # print(myleft,myright)
                        tag= local_tag
                        if(i in aggregation_list):#smaller probe belongs already to a cluster. I keep the local tag
                            myleft = aggregation_list[i]  
                        else:#singleton
                            # if(local_tag==3):
                            #     pass #keep 3 also for singleton. Useful for scoring
                            # else:
                            tag = -1
                        if(j in aggregation_list):#j is the new probe of larger radius
                            myright=aggregation_list[j]
                            if(tag !=-1):
                                # tag=local_tag #This does not enforce connection to singleton.. might geopardize competition between alpha and beta..
                                tag = 4 #if larger probe already belongs to a cluster I force it to tag=4 (pyramidal tag) even if local_tag=3
                                        #I want to enforce a direction ad minimize leakage. 
                                        # I am confident a proper conical shift (local tag=3) is the smallest distance, 
                                        # so it has already been assigned (pyramidal case)
                        else:
                            if(local_tag==3):#larger probe is an isolated probe aligned
                                tag = 3#pass would be equivalent
                            else:
                                tag = -1
                        if(myleft == myright):
                            #They are already in same cluster
                            # if(dendogram[myleft][0].tag==-1):
                            #     if(local_tag==3):
                            #         #Prioritize Conical shift tag to get right signature and direction
                            #         #0 o  o  o   0  
                            #         #0  o o o  0 i (old)  Tag 0 for lateral connections but tag 3 for central one. Is important cluster node appears with tag3
                            #         #     o 3  j new probe (current)
                            #         #NODE UPDATED (left and right are the same so indifferent place holder I use)
                            #         dendogram[myleft][0].tag=3
                            #         dendogram[myleft][0].t_atoms = ids[j-global_shift]
                            #         dendogram[myleft][0].coord = coord[j-global_shift]
                            #         dendogram[myleft][0].r = rp
                            #         continue 
                            #     else:
                            #         continue
                            # else: #tag not updated is already 3 then is ok. Is 0 then I keep it, the connections is pyramidal 
                                # buy with the tip already part of a large cluster.
                                #continue
                                #Tanks to sorting already tag = 3 will be done in priority, since geometrically we expect a truyly tag = 3 is 1 delta far
                                #That is why I skip this
                            continue
                        
                        if(tag==3):
                            refIDS = ids[j-global_shift] #Take from current run
                        else:
                            refIDS = None #give t_atoms list only for the relevant signature, which is the "exact" conical shift
                        
                        #NEW NODE CREATED
                        #ref probe is the larger sphere (current run)
                        dendogram[id_shift+s] = [ClusterNode(id_shift+s,coord[j-global_shift],refIDS, radius = rp,left =dendogram[myleft][0] , right =dendogram[myright][0],tag=tag),d_diff[k]]
                        scluster.append( dendogram[id_shift+s][0])

                        for label in dendogram[id_shift+s][0].pre_order():
                            #if key exists, it simply gets updated
                            aggregation_list[label] = id_shift+s

                        #CHECK FOR POCKETS (preorder also updated self.tag)
                        if((dendogram[id_shift+s][0].tag == 1)and(dendogram[id_shift+s][0].count>=count_Threshold)and (dendogram[id_shift+s][0].r>=rmin)):
                            pcluster.append([dendogram[id_shift+s][0],dendogram[id_shift+s][0].index_btlnk,dendogram[id_shift+s][0].index_Cshift])  
                        ####

                        s+=1
            
                id_shift=len(dendogram)

                # *********************************

                global_shift_old = global_shift
            # *************************************

            n_run+= 1
            global_shift = id_shift
            
            rp_old = rp


        # ********** END OF CLUSTER BUILDING SECTION ************

        # ### INFO
        # print("total number of probes: ", n_total)
        # print("number of linked probes: ", len(aggregation_list))
        self.outFile.write("\n SUMMARY of CLUSTERING PROCESS :total number of probes: "+str(n_total))
        self.outFile.write("\nnumber of linked probes: "+str(len(aggregation_list)))
        ##

        ############# RETRIEVAL STEP 
        # print("\n ============ \n")
        # print("RETRIEVING POCKETS")
        self.outFile.write("\n ============ \n")
        self.outFile.write("RETRIEVING POCKETS")
        # print("alpha = %.1f ; beta = %.1f" %(alpha,beta))
        self.outFile.write("\nalpha = %.1f ; beta = %.1f" %(alpha,beta))
        self.outFile.write("\nShift threshold = "+str(shift_Threshold))
        self.outFile.write("\n Count threshold Pocket = "+str(count_Threshold))
        # print("Minimum radius parent pocket exit= ", rmin)
        self.outFile.write("\n Minimum radius parent pocket exit= " +str(rmin))


        # self.pcluster = sorted(self.pcluster, key = lambda x: x[0].score(weight,len(x['btlnks'])))[::-1]
        if(not isLight):
            try:
                n_nonFilt = len(pcluster)
                pcluster = filtering(pcluster,dendogram)
            except Exception:
                print("Error while filtering pockets.")
                err.value = 2
                err.info = "Error while filtering pockets." 
                return err,[]
            if(not pcluster):
                err.value=1
                err.info = "No pockets found with current parameters."
                return err,[]
            try:
                output = postProcess(pcluster,dendogram)#Retrieve info on mouths, persistence and subpockets
            except Exception:
                print("Error while retrieving mouths and subpockets from pockets.")
                err.value = 2
                err.info = "Error while retrieving mouths and subpockets from pockets."
                return err,[]
        
        else:
            #LIGHT MODE: only pocket nodes are returned. pLarge not used since this make sense to reduce amount of pockets containing rich output
            # print("HERE lightmode")
            try:
                n_nonFilt = len(pcluster)
                pcluster = filtering_light(pcluster)
            except Exception:
                print("Error while filtering pockets.")
                err.value = 2
                err.info = "Error while filtering pockets."
                return err,[]
            if(not pcluster):
                err.value=1
                err.info = "No pockets found with current parameters."
                return err,[]
            output = [{'node':p[0],"subpockets":[]} for p in pcluster] #empty subpocket container

        self.outFile.write("\n---DONE --")
        self.outFile.write("\nInitial number of pockets = " +str(n_nonFilt))
        self.outFile.write("\nNumber of non redundant pockets = " + str(len(pcluster)))
        self.outFile.write("\n \n"+crono.get())
        
        # self.outFile.close()
        # output.load_protein(self.protein)

        self.filteredP = output #formatted pocket container

        return err,self.filteredP

        #####################

    def getRes(self,atomList,getCoord=False,ligandCoord=np.empty(0)):
        """
        Returns residue info of given atoms list (parsed like atom indexes)
        atomList are checked for redundancies (expected, since the spheres can be tangent to identical atoms)
        NEW: mask where the residues where within 4 AA of ligand coord, if given.
        """

        resList=np.asarray(atomList)
        resList = np.unique(resList)
        # resid = [r['resNum'] for r in resMap]
        resMap= self.protein.resMap
        # resname = [r['resName'] for r in resMap]
        # resAtom = [r['resAtom'] for r in resMap]
        proteinCoord = self.protein.atoms
        # print(resList)
        pindex = np.arange(resList.size)
        if (ligandCoord.size>1):
            d,flag = Pdist_C(ligandCoord[:,0:3],proteinCoord[resList,0:3])
            index = np.where(d<=4)[0]
            _lindex,pindex=getIndex(flag,index,resList.shape[0])
            pindex = np.unique(pindex)
            # print(pindex)
            
            # resList = resList[pindex]
            # print(resList)
            # print(resList.size,pindex.size)
        
        mask = np.in1d(resList,resList[pindex])

        if(getCoord):
            resInfo = [(tuple(resMap[r]['coord']+[resMap[r]['radius']]),resMap[r]['atomNumber'],resMap[r]['resAtom'],resMap[r]['resName'],resMap[r]['resNum'],resMap[r]['resChain'],mask[i]) for i,r in enumerate(resList)]
            # resInfo = [(resMap[r]['resAtom'],resMap[r]['resName'],resMap[r]['resNum'],np.append(np.asarray(resMap[r]['coord']),resMap[r]['radius'])) for r in resList]
        else:
            resInfo = [(resMap[r]['atomNumber'],resMap[r]['resAtom'],resMap[r]['resName'],resMap[r]['resNum'],resMap[r]['resChain'],mask[i]) for i,r in enumerate(resList)]

        return resInfo


    def printInfo(self,keep=10,mode = 1, type = "IF", saveSpheres=False,getRes=True,useCNN = False):

        #TODO: If subpocket high in ranking, print it instead of master? High in ranking means that is among first 10...
        err=Error()

        if not self.filteredP:
            try:
                raise FileExistsError ("Cannot save. Filtered pocket list does not exist.\n")
            except FileExistsError:
                err.value= 2
                err.info = "Error in printing of pockets files of "+self.pFilename
                return err

        grid_scale=6
        accTriang = True
        

            # ************* DEBUGGING **************
        #Drop later any sorting, just for debugging..
        self.filteredP=sorted(self.filteredP, key = lambda x: x['node'].count)[::-1] #useless
        
        numberPockets = len(self.filteredP) 
        # count_Threshold = global_module.count_Threshold 
        # weight = count_Threshold
        # pList=sorted(pockets, key = lambda x: x['node'].score(weight,len(x['btlnks'])))[::-1]
        
        mainScore,subScore,featureList,map_direct,map_reverse = getRank(self.filteredP,self.protein.resMap,name=type,mode=mode,structureName =self.pFilename,isPedro=useCNN )

        rankedIndexes = mainScore[0]
        rankedIndexesSub = subScore[0]

        print("%d CANDIDATE POCKETS" %(len(self.filteredP)))

        print(rankedIndexes[:10])
        print(mainScore[1][:10])
        print()
        # print(rankedIndexesSub)
        # print(subScore[1])
        # input("")
        # print("HERE")

        resultFile=open(self.pdbFolder_path+"/"+"output_"+self.pFilename+".txt",'w')
        # resultFile.write("**RANKED POCKETS FOUND FOR "+self.pFilename+" **  \n")
        resultFile.write("%d CANDIDATE POCKETS FOUND FOR "%(len(self.filteredP))+self.pFilename+" **  \n")
        resultFile.write("**RANKED POCKETS**  \n")
        print("First " +str(keep) + " ranked pockets:\n")
        try:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        except Exception:
            resultFile.write("\n ERROR: could not build directory for storing cluster pocket files.")
            err.value= 2
            err.info = "Error in printing of pockets files of "+self.pFilename
            return err

        
        selfContained = set()
        # skippedSet=set()
        #OBS in the following pi is the master pocket index
        k=0 # runs on sorted indexes
        r=1 # runs on ranked positions
        skipped=0
        kept=[]
        while((r<=keep) and (k<rankedIndexes.size)):
            sub=[] #NOTE: In general don't print subs in only one
            print("\nindex = ", rankedIndexes[k])
            pi,si = map_direct[rankedIndexes[k]]
            print(pi,si)
            if(si is not None):
                print("I'm a subpocket")
                if(pi in selfContained):
                    #going to next element (positions do not progress in the ranking) since sub contained in master pocket higher in ranking
                    print("already extpressed by master pocket, skipping")
                    # skippedSet.add(rankedIndexes[k])
                    skipped+=1
                    k+=1
                    continue
                sp = self.filteredP[pi]['subpockets'] #subpocket list of parent pocket
                p = sp[si]
                kept.append(p)
                # ++++ INFO ++++
                print("Pocket number",r)
                resultFile.write("\n\n Pocket number" +str(r))
                print(featureList[map_reverse[(pi,si)]])
                
                ### check correct indexing:
                print(p['node'].count)
                
                ## TODO: change with a simple renaming: the triangulation was already performed for volume computation..
                err = p['node'].buildTriang(triangName='p'+str(r),savePath= self.save_path) #to save triangulation of top ranked pockets 


                if(err.value==1):
                    print("<<WARNING>> Cannot perform triangulation of p%d" %k)
                    err.info =err.info+ "--> Cannot perform triangulation of p%d" %k
                # ** Only for saving methods
                sm = [p['node']] #by definition a subpocket as a single mouth which is its conventional entry
                bm= p['btlnks']
                #************

                # correctedRanking = rank-len(skippedSet)
                n_subs = len(sp)
                if(n_subs==1):
                    print("Single subpocket, master pocket in black list :)")
                    selfContained.add(pi)
                else:
                    ind = map_reverse[(pi,None)]
                    rank = np.where(rankedIndexes == ind)[0][0]+1
                    print("\n **INFO: is subpocket of (absolute rank): p"+str(rank)) 
                # print(ind)
                    # cannot account for final ranking ahead but can point correctly to previously ranked ones. So useless, since if master pocket expressed I'm not giving this

                
            else:
                # Here the ranked pocket could contain subpockets
                print("I'm a parent pocket")
                if(pi in selfContained):
                    # Here only if a SINGLE subpocket was higher in the ranking, this situation is likely to be over redundant
                    print("Single subpocket already expressed in the ranking")
                    # skippedSet.add(rankedIndexes[k])
                    skipped+=1
                    k+=1
                    #PROBLEM <------
                    # **** What about if all its subpockets did hit? I am "conduming" a position in the ranking.. **
                    continue
                

                selfContained.add(pi)
                p = self.filteredP[pi]
                
                # ** Only for saving methods
                sm = [i for i,_d in p['mouths']]
                bm= p['btlnks']
                 
                #***********

                n_subs = len(p['subpockets'])

                # ************** TO CHECK ********
                if(n_subs==1): # promote it if is high in ranking CHECK <-----
                    ind = map_reverse[(pi,0)]
                    rank = np.where(rankedIndexesSub == ind)[0][0] #- len(skippedSet) # without correction to be more exigent
                    print(rank)
                    print(subScore[1][rank])
                    #if(rank<10):
                    if(np.round(subScore[1][rank],2)<0.5): 
                        # so is independent on eventual correction on ranking 
                        #Very much prone to accept subpockets, esecially for EIF where scores are generally lower
                        print("Subpocket elected as better representative of master pocket..")
                        print("SubRank (not accounting for rearrangements):",rank+1)
                        p = p['subpockets'][0]
                        sm = [p['node']]
                        bm= p['btlnks']       
                    else:
                        pass
                #***********************
                kept.append(p)
                # ++++ INFO ++++
                print("Pocket number",r)
                resultFile.write("\n\n Pocket number" +str(r))
                print(featureList[map_reverse[(pi,si)]])
                ### check:
                print(p['node'].count)
                
                # err = p['node'].buildTriang(triangName='p'+str(r),savePath= self.save_path)
                # if(err.value==1):
                #     print("<<WARNING>> Cannot perform triangulation of p%d" %k)
                #     err.info =err.info+ "--> Cannot perform triangulation of p%d" %k
                
                if(n_subs>1):
                    sub = [] # container for printing method [(i['node'],i['btlnks']) for i in p['subpockets']] 
                    print(str(n_subs)+" Subpockets (ranked):")
                    rank=[]
                    for s in range(n_subs):
                        try:
                            rank.append(np.where(rankedIndexesSub==map_reverse[(pi,s)])[0][0])
                        except KeyError:
                            print("Tried to find skipped subpocket")
                            rank.append(numberPockets + 1) # make sure is at the bottom of the rank
                            subScore[1][numberPockets + 1] = 1
                    indS = np.argsort(rank)
                    print("INDS=",indS)
                    print(rank)
                    good_Subs=0
                    for i,sub_i in enumerate(indS):
                        if(subScore[1][rank[sub_i]]<0.5):
                            good_Subs+=1
                            print("Sub ", i, "Absolute Ranking (not accounting for rearrangements ) = ", rank[sub_i] +1 )
                            print((pi,sub_i),"ind="+str(map_reverse[(pi,sub_i)]),'score',subScore[1][rank[sub_i]])
                            print(featureList[map_reverse[(pi,sub_i)]])
                            sp=p['subpockets'][sub_i]['node']
                            err = sp.buildTriang(triangName='sub'+str(r)+'_'+str(sub_i+1),savePath= self.save_path)
                            sub.append((sp,p['subpockets'][sub_i]['btlnks']))
                            if(err.value==1):
                                # print("Increase Self Intersection Grid to compute this pocket!")
                                print("<<WARNING>> Cannot perform triangulation of sub%d_%d" %(r,sub_i))
                                err.info = err.info+ "--> Cannot perform triangulation of sub%d_%d" %(r,sub_i)
                        else:
                            print("BAD SUBPOCKET skipping")
                    if(good_Subs==1):
                        print("Only one good subpocket--> Promoted to master")
                        # print("Consider only the subpocket")
                        p = p['subpockets'][indS[0]]
                        sm = [p['node']]
                        bm= p['btlnks']   
                        sub=[]
                err = p['node'].buildTriang(triangName='p'+str(r),savePath= self.save_path)
                if(err.value==1):
                    print("<<WARNING>> Cannot perform triangulation of p%d" %k)
                    err.info =err.info+ "--> Cannot perform triangulation of p%d" %k
            try:
                if(saveSpheres):
                    saveP(r,self.save_path,p['node'],True,subPockets=sub)
                if(getRes):
                    saveRes(r,self.save_path,p['node'],self.protein.resMap,self.protein.atoms,bmouth = bm,smouth = sm,subPockets = sub)
                    # saveResSimple(r,self.save_path,p['node'],self.protein.resMap)
            except Exception as info:
                err.value=1
                err.info = str(info.args) + " Causing unability to save pocket.pqr files!"
                return err
            k+=1
            r+=1
        print("# skipped :",skipped)
        return err,kept





    def printInfo_old(self,saveSpheres=False,getRes=True,largeP=1,grid_scale=6,accTriang = True):
        from functions import saveP,saveRes,getEntrance
        #KEEP IN MIND: TRAINING DONE AT grid_scale=6
        """
        This must be called after build
        Produces a dedicated file containing detailed inforion on the pockets retrieved.
        If the second argument is true, the pocket spheres are saved in a pqr file.
        """
        #Change with volume? --> in any case all this filtering will be changed with ML result.. not so useful at this stage
        largeThreshold = int(largeP * count_Threshold) #Used for further reduction of pockets returned by the function
        err=Error()
        resultFile=open(self.pdbFolder_path+"/"+"output_"+self.pFilename+".txt",'w')
        resultFile.write("** SUMMARY OF (LARGE) POCKETS FOUND FOR "+self.pFilename+" **  \n")
        
        # self.pocketMode(grid_scale,grid_selfInt,maxProbes_selfInt=200,accTriang=accTriang,gridPerfil=gridPerfil)
        print("Triangulation parameters: grid_scale= %d; Accurate_Triang= %d" %(grid_scale,accTriang))
        # setup_NSInput(self.workingDir+conf,grid_scale=grid_scale,grid_selfInt=2)
        # new_probe(self.workingDir+conf,1.4)#use rp=1.4 to build pocket triangulation

        try:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
        except Exception:
                resultFile.write("\n ERROR: could not build directory for storing cluster pocket files.")
                err.value= 2
                err.info = "Error in printing of pockets files of "+self.pFilename
                return err
        
        if not self.filteredP:
            try:
                raise FileExistsError ("Cannot save. Filtered pocket list does not exist.\n")
            except FileExistsError:
                err.value= 2
                err.info = "Error in printing of pockets files of "+self.pFilename
                return err
        #RANKING
        self.filteredP = sorted(self.filteredP, key = lambda x: x['node'].score(weight,len(x['btlnks'])))[::-1]
        pLarge = list(filter(lambda x: x['node'].count >= largeThreshold, self.filteredP))

        resultFile.write("\n\n Number of non redundant pockets = %d" %len(self.filteredP))
        print("\nNumber of non redundant pockets = %d" %len(self.filteredP))
        resultFile.write("\n Number of large pockets, above %d, are %d " % (largeThreshold,len(pLarge)))
        print("Number of large pockets, above %d, are %d " % (largeThreshold,len(pLarge)))
        s = 1
        for pd in pLarge:
            ######
            print("\n Pocket number",s)
            resultFile.write("\n\n Pocket number" +str(s))

            p=pd['node']
            per = pd['persistence']
            #Compact data containers to save on file
            bm= pd['btlnks']
            sm = pd['mouths']
            
            #Save on file centers pocket 
            try:
                if(saveSpheres):
                    sub = [(i['node'],i['btlnks']) for i in pd['subpockets']] #only functional to priting method
                    saveP(s,self.save_path,p,True,subPockets=sub)
                if(getRes):
                    saveRes(s,self.save_path,p,self.protein.resMap,self.protein.atoms,bmouth = bm,smouth = [i for i,_d in sm],subPockets = sub)
            except Exception as info:
                err.value=1
                err.info = str(info.args) + " Causing unability to save pocket.pqr files!"
                return err
                # print("HERE")
            V,A,err = p.NSVolume(accurate=accTriang,accGridScale=grid_scale,triangName='p'+str(s),savePath= self.save_path) #Produces also triangulated pocket
            # print("here", V,A)
            #also populates new attributes
            
            if(err.value==1):
                # print("Increase Self Intersection Grid to compute this pocket!")
                print("<<WARNING>> Cannot perform triangulation of p%d" %s)
                print("Volume and area computed via semi-analytic method")
                s+=1
                V,A,err = p.volume()
                if(err.value==1):
                    print("<<ERROR>> Cannot compute volume and area. This fields are filled with zeros.")
                    V=0
                    A=0
                    
            
            pd['volume'] = V
            pd['area'] = A
            #####
            #new
            #obs: makes sense only for non subpockets for how its built.. 
            # It could be extended if a similar analysis is done on persistance vectors of subs
            pd['entrances'] = getEntrance(pd['mouths'],pwDist=True)


            r_entrance = 2.6#rmin
            Bradius_list= np.array([i.r for i in bm]) #OBS: filtered. Only smallest radius appearing if multiple probes for same btlnk
            depthList_top = np.asarray([d  for d,r,count in per if r>2.6])#np.asarray([d for _i,d in sm])#I'm keeping it only for the geometrical heuristic tags below..  
            
            if(depthList_top.size==0):
                depthList_top = np.asarray(pd['mouths'][0][1])

                # depthList_top --> average depth of entrances, weigthed by effective radius?
            top_depth = sum([d * r_eff for cm,d,r_eff,r,size in pd['entrances']])/sum([r_eff for cm,d,r_eff,r,size in pd['entrances']])

            
            depthList = np.asarray([d for d,r,count in per])
            average = np.average(depthList)
            average2 = np.average(depthList_top)
            std = np.std(depthList)
            std2 = np.std(depthList_top)
            # Mouthradius_list = np.array([i.r for i,_d in sm])
            Mouthradius_list = np.array([r for d,r,count in per])
            #old
            entrance_list = Mouthradius_list[np.where(Mouthradius_list>=r_entrance)]

            n_differentRad = np.unique(entrance_list).size
            entrance_depth = np.amax(depthList)-average #more than etrance is a sort of measure of the length of the longest axis
            # entrance_depth2 = np.amax(depthList_top)-average2
            entrance_depth2 = top_depth-average #average2 average and not average2 to have equivalent def for subs

            entrance_ramification = abs(entrance_depth - std) #leackage #Should be positively defined.. 
            top_ramification = abs(entrance_depth2 - std) # std instead of std2 for same reason : portability to subs
# NEW comment: it makes more sense to simply use std for entrance ramification (only if prefilter on radius size).. The above is more sort of a measure of an outlier..
#proposal entrance_ramfication = std/(entr_depth ) if ore than one element, 0 otherwise
            shallowness = shift_Threshold -average + shift_Threshold*len(bm)
            
            #If large discepancy between variance and entrance (sup of the variance) the pocket is deep?. 
            # Bottlenecks usually are related to deeper pockets, or at least in a canyon
            # depthScore =entrance_depth - std + len(id_btlnk)#/(np.sum(depthList)+len(id_btlnk)) #smaller flatter
            depthScore = np.amax(depthList_top)*len(bm) - top_ramification
            # depthScore2 = np.amax(depthList)*(Mouthradius_list.size - entrance_list.size) - entrance_ramification
            depthScore2 = (top_ramification - entrance_ramification) # measure of how much exit is protruted 
            #Contribution of intermediate pockets (arbitrely deep and nested) to score without considering bottlenecks in normalization
            ramification = np.sum(depthList)/p.score(w=0) #Inner ramification: how much straight structures WITH POCKET CHARACTERISTICS
            # contribute to the total score. Large: more ramified
            if(round((1-ramification),2)<=0.7):
                        print("RAMIFIED: Consider subpockets\n")
            if(depthScore>0):
                #All this scenarios imply the presence of at least one bottleneck
                if(round(entrance_ramification,2)<=0.4):
                    #REALM OF SMALL LIGANDS?
                    # if(round((1-ramification),2)<0.90):
                    #     print("TUNNEL (invagination with complex network)\n")
                    
                    
                    if(len(sm)==len(entrance_list)):
                        if(len(sm)>=2):
                            if(n_differentRad<2):
                                print("WIDE INVAGINATION")
                            else:
                                print("COMPACT CANYON")
                    elif(len(entrance_list)==1):
                        print("SEALED INVAGINATION")
                    elif(depthScore>np.amax(depthList)and (len(entrance_list))<=(len(sm)//2)and (len(entrance_list)>0)): 
                        print(" CHANNEL ?")
                    elif((len(entrance_list))>(len(sm)//2)and (len(entrance_list)>0)):
                        # if(depthScore>=5*np.amax(depthList)):
                        #         #completely empirical with no logical basis.. better based on radius?
                        #         print("CHANNEL ?")
                        if(n_differentRad>2):
                            print("COMPACT CANYON ")
                        else:
                            print("RAMIFIED INVAGINATION")
                    elif(len(entrance_list)==0):
                        print(" HILLY CANYON")
                    else:
                        print("Compact deep canyon ?\n")     
                else:
                    #Long ligands and proteins (?)
                    if(round(entrance_ramification,2)>=1):
                        if((depthScore>=np.amax(depthList))and (len(entrance_list)<=len(sm)//2)and (len(entrance_list)>0)):
                            print(" VERY WIDE INVAGINATION (OR CHANNEL ?)\n")
                        elif((depthScore>=10*np.amax(depthList))and (len(entrance_list)>0)):
                            if(n_differentRad>2):
                                print("RAMIFIED CANYON")
                            else:
                                print("WIDE CHANNEL (?)")
                        else:
                            print("WIDE CANYON")
                            
                    elif((depthScore>np.amax(depthList))and (len(entrance_list)>0)):
                        print(" DEEP CANYON OR WIDE INVAGINATION\n")
                    else:
                        print("CANYON\n")

            
            elif(depthScore>-0.5):
                if(len(entrance_list)<=len(sm)//2):
                    print("CRATER (buried shallow site) (?)")
                else:
                    print("SHALLOW BINDING SITE \n")

            else:
                if(len(entrance_list)<len(sm)):
                    print("VERY SHALLOW BINDING SITE\n")
                elif(len(entrance_list)<=len(sm)//2):
                    print("WIDE CRATER (?)")
                else:
                    print("HILL")
            
            print("MAIN SCORE = ", p.score(weight,len(bm)))
            resultFile.write("\nMAIN SCORE = " + str(p.score(weight,len(bm))))

            print("VOLUME = ", pd['volume'])
            resultFile.write("\n VOLUME = "+ str(pd['volume']))
            print("AREA = ", pd['area'])
            resultFile.write("\n AREA = "+ str(pd['area']))

            print("** NEW measures of compactness:")
            magicRatio = V/pd['node'].count #V/pd['node'].volume(consider_overlap = False)[0]
            HW_score = np.pi**(1./3.)*(6*V)**(2./3.)/A
            print("HW_score= ", HW_score)
            print("magicRatio= ", magicRatio)
            print("--")

            print("Clustered Entrances: ", len(pd['entrances']))
            print("Entrance List:", pd['entrances'])
            resultFile.write("\n Clustered Entrances: "+ str(len(pd['entrances'])))
            print("# elements: ",p.count)
            resultFile.write("\n# elements: "+str(p.count))
            print(" All mouth radii:", Mouthradius_list)
            resultFile.write("\nAll mouth radii:" +str(Mouthradius_list))
            print('Entrances radii:', entrance_list)
            resultFile.write("\n Entrances radii:" +str(entrance_list))
            print("# entrances (top probe of r>%.1f )= %d" %(r_entrance,len(entrance_list)))
            resultFile.write("\n# entrances (top probe of r>%.1f )= %d" %(r_entrance,len(entrance_list)))
            print("# Mouth vector size (size requirement): ",len(sm) )
            resultFile.write("\n# Mouth vector size (size requirement): "+str(len(sm)) )
            # print("Overall persistence vector: \n", persistance)
            print("# Persistence vector size (all shifts): ",len(per) )
            resultFile.write("\n# Persistence vector size (all shifts): "+str(len(per)) )
            print("# bottlenecks: ",len(bm) )
            resultFile.write("\n# bottlenecks: " +str(len(bm)))
            print ("Max depth (for mouths) = ", np.amax(depthList) )
            resultFile.write("\nMax depth (for mouths) = " + str(np.amax(depthList)))
            print ("Average depth= ", average) #many termination by large radii suggest I'm often going "out"
            resultFile.write("\nAverage depth= " +str(average))
            #print ("Entrance profondity = ", entrance_depth )
            print("Botlleneck radius list:", Bradius_list)
            resultFile.write("\nBotlleneck radius list:" + str(Bradius_list))
            print("Global ramification= ",entrance_ramification)
            resultFile.write("\nGlobal ramificatio= "+str(entrance_ramification))
            print("Compactness score = %.1f %%" %((1-ramification)*100))
            resultFile.write("\nCompactness score = %.1f %%" %((1-ramification)*100))
            print ("Depth score (How deep/buried is a structure)= ", depthScore)#more immune to pocekt size limitations and more robust with notion of what is an exit
            resultFile.write("\nDepth score (How deep/buried is a structure)= " +str(depthScore))
            print("Alternative depth score= ", shallowness)
            resultFile.write("\nAlternative depth score= " +str(shallowness))
            print("Leackage score = ", -depthScore2)
            resultFile.write("\nLeackage score= "+ str(-depthScore2))

            print("# subPockets: ", len(pd['subpockets']))
            resultFile.write("\n# subPockets: " +str(len(pd['subpockets'])))
            for i,sb in enumerate(pd['subpockets']):
                V,A,err = sb['node'].NSVolume(accurate=accTriang,accGridScale=grid_scale,triangName='sub'+str(s)+'_'+str(i+1),savePath= self.save_path) #Produces also triangulated pocket
                sb['volume'] = V
                sb['area'] = A
                per = sb['persistence']
                magicRatio = V/sb['node'].count #V/sb['node'].volume(consider_overlap = False)[0]
                HW_score = np.pi**(1./3.)*(6*V)**(2./3.)/A
                # depthList_top = np.asarray([d for d,r,count in per if r>2.6 ])
                top_depth = sb['depth']
                depthList = np.asarray([d for d,r,count in per])
                average = np.average(depthList)
                # average2 = np.average(depthList_top)
                std = np.std(depthList)
                # std2 = np.std(depthList_top)
                entrance_depth = np.amax(depthList)-average #more than etrance is a sort of measure of the length of the longest axis
                entrance_depth2 = top_depth-average 
                entrance_ramification = abs(entrance_depth - std)  
                top_ramification = abs(entrance_depth2 - std)
                depthScore = top_ramification - entrance_ramification
                shallowness = shift_Threshold -average
                # sb['entrances'] = getEntrance(sb['persistence'],pwDist=True)
                if(err.value==1):
                    # err.value=1
                    # err.info = errline
                    print("Increase Self Intersection Grid to compute this subpocket!")
                    continue
                
                print("\n \t Subpocket %d SIZE= %d, VOLUME = %.2f, AREA = %.2f, DEPTH and radius: %d,  %.1f, SCORE = %d, BTLNKS= %d:" 
                %(i+1,sb['node'].count,sb['volume'],sb['area'],sb['depth'],sb['rtop'],sb['node'].score(weight,len(sb['btlnks'])),len(sb['btlnks'])))
                resultFile.write("\n \t Subpocket %d SIZE= %d, DEPTH = %d  SCORE = %d, BTLNKS= %d:" 
                %(i,sb['node'].count,sb['depth'],sb['node'].score(weight,len(sb['btlnks'])),len(sb['btlnks'])))
                print("\t magicRatio=", magicRatio)
                print("\t HW_score= ", HW_score)
                print("\t Leackage score=",-depthScore)
                # print("\t DepthScore2=",shallowness)
            #Depth here is a sort of burying measure. Higher the number more likely the subpocket is a main entrance
            s+=1
        resultFile.close()
        return err,pLarge
            



    ################
    #############################

    ############## FILTERING FUNCTIONS #############
############################## FILTERING FUNCTION ########################################

def filtering(pcluster,dendogram):
    '''
    Input: dendogram and gathering parameters
    Return: 
        NEW: pocket cluster with persistence and bottlenecks. 
            Proper pocket mouths and subpockets can be filtered outside in a separate routine
        OLD:pcluster and bclusters filtered 
            list of sub-pockets only for pcluster (subpockets are not defined for bclusters)

            sub-pockets are filtered for removing nested ones and keeping only the SMALLEST. Nested pocket shift do not exist by definition.
            Bottleneck mouths however could be present.
            MOUTHS --> index_btlnk, index_Cshift


    mouths filtering : Bottleneck redundancy are solved keeping as unique bottleneck the one relevant to the SMALLEST probe
                      Pseudo-exits redundancy "" keep the one relevant to the LARGEST probe.
                      Keeping track of depth of the redundancy
                      Distinguish between persistence list and proper mouths (size criterion also)

    '''

    #IDEA: Complementary between sub pockets and large pocket defines the riverbed!!

    
    #TODO: Check if doable using set()? Should be faster.. The problem is that I need to not jump elements when deleting..
    #HP: sort before with size. Maybe sorting is fast. Then I can delete only last element of list pop() method, which allows to use set..
    #Problem set wants unashable elements, so I have to check if doable..

    found = 0
    s = 0
    while s < (len(pcluster)):
        n = pcluster[s][0].count
        elements = np.asarray(pcluster[s][0].pre_order())
        #for nested_p in set([p[0] for p in pcluster]):
        for nested_p in pcluster:
            if(nested_p[0] is pcluster[s][0]):
                pass
            else:
                if(np.intersect1d(elements,np.asarray(nested_p[0].pre_order()),assume_unique=True).any()):
                    if(n<nested_p[0].count):
                        del pcluster[s]
                        found = 1
                        break
                    else:
                        pass
        if(found):
            found = 0
            continue #back to main loop without updating s (indexes of new list are shifted backwards)
        #Here on pcluster contains no nested pockets, no redundancies (however mouth list accounts for nested pockets)
        #**************************
        #Bottleneck mouths filtering
        index_btlnk = pcluster[s][1]
        index_Cshift = pcluster[s][2] 
        if(len(index_btlnk)>1):
            atomList=[]     
            rList =[]                                                                                                                        
            for k in index_btlnk: 
                atomList.append(dendogram[k][0].t_atoms)
                rList.append(dendogram[k][0].r)
            atomList = np.asarray(atomList)
            rList = np.asarray(rList)
            atomList.sort(axis=1)
            idx_sort = np.lexsort((atomList[:,2],atomList[:,1],atomList[:,0]))
            atomList = atomList[idx_sort]
            u,idx_start,count = np.unique(atomList,return_index =True,return_counts = True,axis = 0)
            if((count>1).any()):
                # n_redundant= np.sum(count[count>1])
                #print("Found ",n_redundant, "redundant bottlenecks in ", pcluster[s][0].id)
                res  =np.split(idx_sort, np.sort(idx_start))
                res = list(filter(lambda x: x.size > 0, res))
                good_indx=[]
                for k in res:
                    good_indx.append(index_btlnk[k[np.argsort(rList[k])[0]]])
                pcluster[s][1] = good_indx
            else:
                #No redundancies found
                pass
        #SAME FOR Cshifs (but reverse sorting with respect to radius for atom list):
        if(len(index_Cshift)>1):
            atomList=[]     
            rList =[]                                                                                                         
            for k,_nt3 in index_Cshift: 
                atomList.append(dendogram[k][0].t_atoms)
                rList.append(dendogram[k][0].r)
            atomList = np.asarray(atomList)
            rList = np.asarray(rList)
            atomList.sort(axis=1) #Ensures rows are comparable among them(same ordering, since np.unique is sensitive to order within rows)
            idx_sort = np.lexsort((atomList[:,2],atomList[:,1],atomList[:,0]))#Puts same rows close by, necessary fo the use of "res"
            atomList = atomList[idx_sort]
            
            u,idx_start,count = np.unique(atomList,return_index =True,return_counts = True,axis = 0)
            if((count>1).any()):
                n_redundant= np.sum(count[count>1])
                res  =np.split(idx_sort, np.sort(idx_start))
                res = list(filter(lambda x: x.size > 0, res))
                good_indx=[]
                for k in res:
                    #Keep the one associated to larger rp and trace redundancies in the second argument of the tuple
                    good_indx.append((index_Cshift[k[np.argsort(-rList[k])[0]]][0],index_Cshift[k[np.argsort(-rList[k])[0]]][1]))
                pcluster[s][2] = good_indx
                
        else:
            #No redundancies found
            pass
        s+=1
    
    return pcluster;
##############################

def filtering_light(pcluster):
    '''
    Does not filter mouths and bottlenecks.
    '''

    #Filtering redundant (nested) pockets

    found = 0
    s = 0
    while s < (len(pcluster)):
        n = pcluster[s][0].count
        elements = np.asarray(pcluster[s][0].pre_order())
        #for nested_p in set([p[0] for p in pcluster]):
        for nested_p in pcluster:
            if(nested_p[0] is pcluster[s][0]):
                pass
            else:
                if(np.intersect1d(elements,np.asarray(nested_p[0].pre_order()),assume_unique=True).any()):
                    if(n<nested_p[0].count):
                        del pcluster[s]
                        found = 1
                        break
                    else:
                        pass
        if(found):
            found = 0
            continue #back to main loop without updating s (indexes of new list are shifted backwards)
        #Here on pcluster contains no nested pockets, no redundancies (however mouth list accounts for nested pockets)
        s+=1

    return pcluster;
##############



def postProcess(pocketList,dendogram):
    """
    Function building persistence vector, mouth list and subpockets of passed pockets. 
    The pseudo mouth list OF INPUT pocket (NOT other clusters, such as subs..) 
    must alredy have been filtered for redundancies
    Returns a list of dictionaries with pocket data
    """
    # print("\n--RETRIEVING MOUTHS AND SUBPOCKETS of large Pockets passed")
    richFormat = []
    for p,id_btlnk,id_Cshift in pocketList:
        persistence = np.asarray([(d,dendogram[i][0].r,dendogram[i][0].count) for i,d in id_Cshift])
        #Conventional pocket mouths have a size requirement over the persistence list-->defines subpockets
        mouthList = list(filter(lambda x: dendogram[x[0]][0].count>=count_Threshold ,id_Cshift))
        #NOTE: mouthList[0] is the actual mouth defined as the latest created node
        subPlist = [[m] for m in list(filter(lambda x: x[0]!=p.id, mouthList)) ] 
        k = 0
        found = 0
        while k < len(subPlist):
            node = dendogram[subPlist[k][0][0]][0]
            elements = np.asarray(node.pre_order())
            n = node.count
            #for nested_s,_ in subPlist:
            for i in subPlist:
                nested_s = dendogram[i[0][0]][0]
                if(np.intersect1d(elements,np.asarray(nested_s.pre_order())).any()):
                    if(n>nested_s.count):#keep smallest
                        del subPlist[k] 
                        found = 1
                        break
                    else:
                        pass
            if(found):
                found = 0
                continue
            #here on, unique subpockets 
            #I use btlnk list rather than tag since already filtered for redundancies.
            btlnk_inSub=[]
            for b in id_btlnk:
                b_elements = np.asarray(dendogram[b][0].pre_order())
                nb= dendogram[b][0].count
                if(np.intersect1d(elements,b_elements).any()):
                    if(nb<n):
                        btlnk_inSub.append(b)
                        #want to ensure bottleneck node is not a parent node
                    else:
                        pass 
            
            sub_pers = np.asarray([(d,dendogram[i][0].r,dendogram[i][0].count) for i,d in dendogram[subPlist[k][0][0]][0].index_Cshift])
            #to avoid refiltering keep only index_Cshifts present in the original set, which are already filtered
            sub_pers = sub_pers[(sub_pers[:, None] == persistence).all(-1).any(-1)]
            # print('here')
            subPlist[k].append(btlnk_inSub)
            subPlist[k].append(sub_pers)
            k+=1
        # print('here')#kth subpocket within sth pcluster
        subPlist = sorted(subPlist, key = lambda x: dendogram[x[0][0]][0].score(weight,len(x[1])))[::-1]#sort according to score
        
        globalSubList =[{'node':dendogram[i[0][0]][0],'depth':i[0][1],'rtop':dendogram[i[0][0]][0].r,
        'btlnks':[dendogram[j][0]for j in i[1]],"btlnk_info":[(dendogram[j][0].r,dendogram[j][0].count)for j in i[1]],'persistence':i[2],
        "aggregations":[(dendogram[j][0],label,dendogram[j][0].r,dendogram[j][0].left.count,dendogram[j][0].right.count) for label,j in dendogram[i[0][0]][0].get_aggregations()]} for i in subPlist]

        bmouths = [dendogram[i][0] for i in id_btlnk]
        smouths = [(dendogram[i][0],d) for i,d in mouthList] #Related to conventional def of pocket
        
        all_mouths = list(filter(lambda x: dendogram[x[0]][0].r>=rmin_entrance ,id_Cshift)) #Related to geometrical constraints
        amouths = [(dendogram[i][0],d) for i,d in all_mouths]

        smouths = sorted(smouths, key = lambda x: (x[0].r,x[0].count))[::-1] #sort based on radius and size so that first element is the main mouth
        dt = {"node": p, "subpockets": globalSubList, "persistence": persistence,
        "btlnk_info":[(dendogram[i][0].r,dendogram[i][0].count) for i in id_btlnk], "btlnks":bmouths,"mouths":smouths,"all_mouths":amouths,
        "aggregations":[(dendogram[i][0],label,dendogram[i][0].r,dendogram[i][0].left.count,dendogram[i][0].right.count) for label,i in p.get_aggregations()]}
        richFormat.append(dt)
    # print("\n DONE ")
    return richFormat

###########
#############################

