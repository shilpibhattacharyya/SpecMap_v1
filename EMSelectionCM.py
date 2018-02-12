import numpy as np
import itertools
import pandas as pd
import operator
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
class GenerateSpectrumMap:
    '''
    Given a set of Tx-locations and Tx-powers, and params related to wireless propagation model,
    generate:
        i) all permutations(ON/OFF of transmitters) of maps
        ii) training data for clustering algorithms
    '''
    def __init__(self,
                    max_x_meter,
                    max_y_meter,
                    tx_power_dBm,
                    tx_loc,
                    n = 2.0,
                    lmda_meter = 0.3,
                    d_0_meter = 1.0,
                    sigma_sq_db = 5.0,
                    noise_floor_dB = -60.0,
                    ksensorsloc=[],
                    kprimesensorsloc=[]
                 ):
        '''
        set the params
        '''
        self.max_x_meter = max_x_meter
        self.max_y_meter = max_y_meter
        self.tx_power_dBm = tx_power_dBm
        self.tx_loc = tx_loc
        self.n = n
        self.lmbda_meter = lmda_meter
        self.d_0_meter =  d_0_meter
        self.sigma_db = np.sqrt(sigma_sq_db)

        self.k_dB = - 20.0 * np.log10(4.0 * np.pi * self.d_0_meter / self.lmbda_meter)
        self.noise_floor_dB = noise_floor_dB
        self.ksensorsloc = []
        self.kprimesensorsloc = []
        np.random.seed(1009993)

    def avgPairwiseMapError(self,maps_A_dBm, maps_B_dBm):
        ''' 
        compute bipartite graph between maps_A_dBm, and maps_B_dBm based on minimum error
        and generate avg error
        :param maps_A_dBm:
        :param maps_B_dBm:
        :return: mapA indices, corresponding mapB indices, avg_map_error
        '''
        error_matrix = np.zeros( shape = ( len(maps_A_dBm), len(maps_B_dBm) ) ) 
        for i,cur_map_A in enumerate(maps_A_dBm):
            for j, cur_map_B in enumerate(maps_B_dBm):
                error_matrix[i][j] = self.computeMapAccuracy(cur_map_A, cur_map_B)
        #if dimensions are equal, just associate row-wise mins
        min_err_row_indx, min_err_col_indx = linear_sum_assignment(error_matrix)

        #if dimensions are unequal, re-run association
        new_min_error_row_indx = []
        new_min_error_col_indx = []
        if len(maps_A_dBm) > len(maps_B_dBm): #unassociated rows there
            unassociated_rows = list(  set(range( len(maps_A_dBm) )) - set(min_err_row_indx)  )
            new_error_matrix = error_matrix[ unassociated_rows , :]
            _,  new_min_error_col_indx = linear_sum_assignment(new_error_matrix)
            new_min_error_row_indx = unassociated_rows
        elif len(maps_A_dBm) < len(maps_B_dBm): #unassociated cols there
            unassociated_cols = list(  set(range( len(maps_B_dBm) ) ) - set(min_err_col_indx)  )
            new_error_matrix = error_matrix[ : , unassociated_cols].T
            _, new_min_error_row_indx = linear_sum_assignment(new_error_matrix)
            new_min_error_col_indx = unassociated_cols
        map_A_assoc_indices = list(min_err_row_indx)+list(new_min_error_row_indx)
        map_B_assoc_indices = list(min_err_col_indx) + list(new_min_error_col_indx)

        avg_map_error =  np.average(  error_matrix[map_A_assoc_indices, map_B_assoc_indices] )
        return map_A_assoc_indices, map_B_assoc_indices, avg_map_error

    def generateDerivedMaps(self, max_x, max_y, predicted_labels, x_indices, y_indices, vals):
        ''' 
        tasks:
            i) combine all clustered samples to create map for each component
        :return:
        '''
        derivedMaps = []
        max_predicted_label = max( predicted_labels )
        for cur_label in range(max_predicted_label+1):
            # #check if there is this label in the predicted label list at all
            # if cur_label not in max_predicted_label:
            #     #self.derivedMaps.append(None)
            #     continue
            cur_map_val = np.zeros(shape=(max_x, max_y ))
            cur_map_count = np.zeros(shape=(max_x, max_y ))

            #-----first aggregate the values of available in the samples of the same cluster
            for k, l in enumerate(predicted_labels):
                if l==cur_label:
                    for x_indx, y_indx, val in zip(x_indices[k],
                                                   y_indices[k],
                                                   vals[k]):
                        cur_map_val[x_indx][y_indx] += val 
                        cur_map_count[x_indx][y_indx] += 1
            cur_map_val = np.divide(cur_map_val, cur_map_count, where = cur_map_count > 0.0)

            #-----interpolate the rest of the locations--------------------------
            c = 0
            x_has_val, y_has_val = np.nonzero(cur_map_count)
            val_at_xy = cur_map_val[ x_has_val, y_has_val ]
            for i in range(max_x):
                for j in range(max_y):
                    if cur_map_count[i][j] <= 0.0:
                        interpolated_val = self.idw(xs = x_has_val,
                                                    ys = y_has_val,
                                                    vals= val_at_xy,
                                                    xi=i, yi=j)
                        cur_map_val[i][j] = interpolated_val
                        c= c+1
            #finally save the map
            derivedMaps.append(cur_map_val)
            #print('in generateDerivedMaps', c)

        return derivedMaps

    def computeMapAccuracy(self, mapA_dBm, mapB_dBm):
        ''' 
        convert dBm to miliwatt and then take pairwise percent diff, here, mapA_dBm is the original map to base %-error
        :return:
        '''
        mapA_mW = np.power(10.0, mapA_dBm / 10.0)
        mapB_mW = np.power(10.0, mapB_dBm / 10.0)

        mapA_mW[mapA_dBm == 0.0] = 0.000001 #to avoid div-by-zero

        error_A_B = (np.abs(mapA_mW - mapB_mW)) / mapA_mW
        avg_error = np.average(error_A_B)
        return avg_error

    
    def idw(self, xs, ys, vals, xi, yi):
        '''
            :param xs:
            :param ys:
            :param vals:
            :param xi:
            :param yi:
            :return:
            '''
        sum_of_weights = 0.0
        sum_of_weight_times_val = 0.0
        for x,y,z in zip(xs,ys,vals):
            dist_sq = (x - xi)**2 + (y - yi)**2
            if dist_sq <=0.0:
                dist_sq = 0.0000001
            w = 1.0/dist_sq
            sum_of_weights += w
            sum_of_weight_times_val += z*w
        interp_val = sum_of_weight_times_val/sum_of_weights
        return interp_val



    def generateIndividualMap(self):
        '''
        generates a map for each Tx
        :return:
        '''

        self.ind_map = []
        self.all_maps = []
        print('self.tx_power_dBm.shape[0]: ', self.tx_power_dBm)
        for i in range(self.tx_power_dBm.shape[0]):
            tx_x, tx_y = self.tx_loc[i][0], self.tx_loc[i][1]

            x_vals = np.arange(0, self.max_x_meter+1, self.d_0_meter)
            y_vals = np.arange(0, self.max_y_meter+1, self.d_0_meter)

            x_grid, y_grid = np.meshgrid(x_vals, y_vals, sparse=False, indexing='ij')
            #print('x_grid: ', x_grid)
            #print('y_grid: ', y_grid)
            dist_sq_map = (  (x_grid - tx_x)**2.0 + (y_grid - tx_y)**2.0 )

            path_loss = 5*self.n*np.log10( dist_sq_map/self.d_0_meter**2.0, where = dist_sq_map > 0.0)
            path_loss[dist_sq_map <= 0.0] = 0.0
            cur_map = self.tx_power_dBm[i] - path_loss
            self.ind_map.append(cur_map)
    #self.displayMaps(self.ind_map)

    def get_map_grid_x(self):
        return self.ind_map[0].shape[0]

    def get_map_grid_y(self):
        return self.ind_map[0].shape[1]

    def combineMap(self, indexList):
        '''

        :param indexList:
        :return:
        '''
        cur_map_mW = np.zeros_like( self.ind_map[0] )
        for indx in indexList:
            cur_map_mW +=  np.power(10, self.ind_map[indx]/10.0)

        cur_map_dB = 10*np.log10(cur_map_mW, where = cur_map_mW>0.0)
        cur_map_dB[cur_map_mW<=0.0] = self.noise_floor_dB

        self.all_maps.append(cur_map_dB)
    

    def generateAllCombinationMap(self):
        '''

        :return:
        '''
        all_indx = range(0, len( self.ind_map  )  )
        for l in range(1, len(all_indx) + 1):
            for comb in itertools.combinations(all_indx, l):
                self.combineMap(list(comb))
        #print('self.allmaps shape: ', len(self.all_maps))
        #self.displayMaps(self.all_maps, 2)
        

    def displayMaps(self, map_list, figFilename, n_rows = 1):
        fig = plt.figure()
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(n_rows, int(  np.round( len(map_list)/n_rows) )),
                        axes_pad=0.01,
                        share_all=True,
                        label_mode="L",
                        cbar_location="right",
                        cbar_mode="single",
                        )

        for cur_map, ax in zip(map_list, grid):
            im = ax.imshow(cur_map, vmin=self.noise_floor_dB, vmax=0.0)
        grid.cbar_axes[0].colorbar(im)
        #plt.show(block = False)
        plt.savefig(figFilename)

    def generateData(self, n_sample, add_noise, flag, numsensors):
        '''
        randomly generate sample_count samples each with a dim_ratio of dimensions
        return the sample array
        :param dim_ratio:
        :return:
        '''
        self.training_data_vals = []
        self.training_data_x_indx = []
        self.training_data_y_indx = []
        self.training_data_labels = [] #index to self.all_maps
        
        total_maps = len(self.all_maps)
        map_max_x, map_max_y = self.all_maps[0].shape
        all_loc_indx = list(np.ndindex(map_max_x, map_max_y))
        total_indices = len(all_loc_indx)
        indices_to_be_chosen = numsensors#max(1, int(np.round(dim_ratio*map_max_x*map_max_y)))
        print("number of sensors(positions which will with values): ", numsensors)

        for i in np.arange(n_sample):
            map_indx = np.random.choice(total_maps, 1)[0]
            indx_to_locs = np.random.choice(total_indices, indices_to_be_chosen, replace = False)
            chosen_indices = sorted([all_loc_indx[cindx] for cindx in indx_to_locs])
            if flag:
                self.ksensorsloc.append(chosen_indices)
            elif chosen_indices not in self.ksensorsloc :
                self.kprimesensorsloc.append(chosen_indices)
                chosen_indices=[x for x in chosen_indices if x not in self.ksensorsloc]
            
            #print('chosen_indices: ', chosen_indices)
            x_indx, y_indx = zip(*chosen_indices)
	    #print('x_indx length: ', len(x_indx))
            chosen_signal_vals = self.all_maps[map_indx][x_indx , y_indx]
            if add_noise:
                shadowing_vals = np.random.normal(loc = 0.0,
                                                  scale = self.sigma_db,
                                                  size = len(chosen_signal_vals)
                                                 )
                chosen_signal_vals +=  shadowing_vals
            self.training_data_labels.append(map_indx)
            self.training_data_vals.append(chosen_signal_vals)
            self.training_data_x_indx.append(x_indx)
            self.training_data_y_indx.append(y_indx)


    def prettyPrintSamples(self):
        '''
        mostly for debug
        :return:
        '''
        counter = 1
        for x_indices, y_indices, vals, label in zip(self.training_data_x_indx,
                                               self.training_data_y_indx,
                                               self.training_data_vals,
                                                self.training_data_labels
                                               ):
            print( "Training Sample# ",counter, " map indx: ",label)
            counter += 1
            for x_indx, y_indx, val in zip(x_indices, y_indices, vals):
                print("\t",x_indx,",",y_indx," : ",val)

    def displayTrainingDataMap(self, map_indx):
        '''
        displays the heatmap for map_index training data
        :param map_indx:
        :return:
        '''
        cur_map =  np.empty_like(self.all_maps[0])
        cur_map[:] = np.nan
        for x_indx, y_indx, val in zip(self.training_data_x_indx[map_indx],
                                       self.training_data_y_indx[map_indx],
                                       self.training_data_vals[map_indx]
                                       ):
            cur_map[ x_indx,y_indx ] = val
        self.displayMaps([self.all_maps[ self.training_data_labels[map_indx]], cur_map])
    
    
    def interpolateMissingData(self, pca, x_indices, y_indices, vals, labels):
        '''
            handle missing values by interpolation
            :return:
            '''
	self.cur_training_matrix = None
	#find set of all unique features as tuple-list of coordinates
	self.index_tuples = []
	#print('self.x_indices shape: ', len(x_indices))
	#print('self.y_indices shape: ', len(y_indices))
	for x_indx, y_indx in zip(x_indices, y_indices):
	    self.index_tuples.extend(zip(x_indx, y_indx))
	    #print('index_tuples shape: ', len(self.index_tuples))
	    self.index_tuples = sorted(list(set(self.index_tuples)))

		
	#now create a training matrix for each datapoint
	for x_indx, y_indx, cur_vals, label in zip( x_indices,
						   y_indices,
						   vals,
						   labels
						   ):
	    cur_vector = []
	    #scan for all features, if the value exists, copy, if not interpolate
	    cur_indices = zip(x_indx, y_indx)
	    cur_indices_list = list(cur_indices)
	    #print('cur indices:',list(cur_indices))
	    o = 0
	    for ix, iy in self.index_tuples:
		if (ix, iy) in cur_indices_list:
		    cur_val = cur_vals[cur_indices_list.index((ix, iy))]
		    cur_vector.append(cur_val)
		#else interpolate
		else:
		    #o+=1
		    interpolated_val = self.idw(xs = x_indx,
						ys = y_indx,
						vals=cur_vals,
						xi = ix,
						yi = iy)
		    cur_vector.append(interpolated_val)

	    #print('hi', o)
	    if self.cur_training_matrix is None:
		self.cur_training_matrix = np.array(cur_vector)
	    else:
		self.cur_training_matrix = np.vstack( (self.cur_training_matrix,
							       np.array(cur_vector))
							     )
		
	#else:	
	if pca > 0:
		print self.cur_training_matrix.shape
		pca = PCA(n_components=pca)
		self.cur_training_matrix =  pca.fit_transform(self.cur_training_matrix)
		print "Feature Reduction for ",pca," sensors: ", pca.n_components_ ," / ",len(self.index_tuples), "variance: ", pca.explained_variance_ratio_
		self.var = pca.explained_variance_ratio_
		print self.cur_training_matrix.shape
		#print self.cur_training_matrix

	
	#print('cur training matrix size: ' , len(cur_indices_list))
	return self.cur_training_matrix


    def interpolateMissingDataK(self, pca, x_indices, y_indices, vals, labels, train_matrix):
        '''
            handle missing values by interpolation
            :return:
            '''
	self.cur_training_matrix = train_matrix
	#else:  
        if pca > 0:
                print (np.array(self.cur_training_matrix)).shape
                pca = PCA(n_components=pca)
                self.cur_training_matrix =  pca.fit_transform(self.cur_training_matrix)
                print "Feature Reduction for ",pca," sensors: ", pca.n_components_ ," / ",len(self.index_tuples), "variance: ", pca.explained_variance_ratio_
		self.var = pca.explained_variance_ratio_
                print self.cur_training_matrix.shape
                #print self.cur_training_matrix


        #print('cur training matrix size: ' , len(cur_indices_list))
        return self.cur_training_matrix



    def runGMMClusteringForKnownComponents(self, n_components, covariance_type = 'full'):
        '''
            
            :return: ARI value
            '''
                #self.cur_n_component = n_components
	self.cur_training_matrix = np.array(self.cur_training_matrix)
        print('self.cur_training_matrix new shape:', self.cur_training_matrix.shape) 
        gmm = GaussianMixture( n_components = n_components, covariance_type = covariance_type).fit(self.cur_training_matrix)
        predicted_labels = gmm.predict(self.cur_training_matrix)
        probs = gmm.predict_proba(self.cur_training_matrix)
	var = gmm.covariances_
        cur_ari = adjusted_rand_score(self.training_data_labels, predicted_labels)
        cur_aic = gmm.aic(self.cur_training_matrix)
        cur_bic = gmm.bic(self.cur_training_matrix)
        return cur_bic, cur_aic, cur_ari, predicted_labels, probs, var#sorted(np.sum(gmm.predict_proba(self.cur_training_matrix), axis=1), reverse=True)[0]

    def runGMMClustering(self, n_component_list, covariance_type = 'full'):
        '''
            based on bic find the best # of components
            :param covariance_type:
            :return:
            '''
        
        min_comp, min_bic, min_aic, min_ari, min_predicted_labels = None, \
                                                                float('inf'), \
                                                                float('inf'), \
                                                                float('inf'), \
                                                                []  

        kl =  1
        kh =  n_component_list
        low = 1 
        high = n_component_list

        while kh > kl: 
                print('kh , kl: ', kh, kl) 
                cur_bic_kl, cur_aic_kl, cur_ari_kl, predicted_labels_kl, cur_cm_kl, cur_var_kl = self.runGMMClusteringForKnownComponents(kl,
                                                                                        covariance_type)


                cur_bic_kh, cur_aic_kh, cur_ari_kh, predicted_labels_kh, cur_cm_kh, cur_var_kh = self.runGMMClusteringForKnownComponents(kh,                                                                                                                                    covariance_type)

                if cur_bic_kl < cur_bic_kh: #and cur_bic_kl < min_bic:
                    kh = (kh + kl)/2
                    min_comp, min_bic, min_aic, min_ari, min_predicted_labels, min_cm_, min_var_ = kh, \
                                                                  cur_bic_kh, \
                                                                  cur_aic_kh, \
                                                                  cur_ari_kh, \
                                                                  predicted_labels_kh, cur_cm_kh, cur_var_kh 
                    kl+=1
                elif cur_bic_kh < cur_bic_kl:# and cur_bic_kh < min_bic:                                                                 
                    kl = (kh + kl)/2
                    min_comp, min_bic, min_aic, min_ari, min_predicted_labels, min_cm_, min_var_ = kl, \
                                                                  cur_bic_kl, \
                                                                  cur_aic_kl, \
                                                                  cur_ari_kl, \
                                                                  predicted_labels_kl, cur_cm_kl, cur_var_kl

                    kh-=1

                print "DEBUG: cur_bic_kl: #components, BIC, AIC, ARI: ",   cur_bic_kl, cur_aic_kl, cur_ari_kl
                print "DEBUG: cur_bic_kh: #components, BIC, AIC, ARI: ",   cur_bic_kh, cur_aic_kh, cur_ari_kh
                print "DEBUG: min: #components, BIC, AIC, ARI: ",   min_comp, min_bic, min_aic, min_ari
        #re-run gmm for the min-bic case if compoenet > 1

        self.gmm_min_comp, self.gmm_min_bic, self.gmm_min_aic, self.gmm_predicted_labels = \
                 min_comp,          min_bic,          min_aic,      min_predicted_labels

        return min_comp, min_bic, min_aic, min_ari, min_predicted_labels, min_cm_, min_var_

	
	#min_comp, min_bic, min_aic, max_ari, min_predicted_labels, cm_ = None, 0.0, 0.0, 0.0, 0.0,[]
        #for c in n_component_list:
        #    cur_bic, cur_aic, cur_ari, predicted_labels, cm, var = self.runGMMClusteringForKnownComponents(n_components =c, covariance_type=covariance_type)
        #    if cur_ari > max_ari:
        #        min_comp, min_bic, min_aic, max_ari, min_predicted_labels, cm_, var_ = c, cur_bic, cur_aic, cur_ari, predicted_labels, cm, var
    
        #    print("DEBUG: #components, BIC, AIC, ARI: ", c, cur_bic, cur_aic, cur_ari)
    
        ##re-run gmm for the min-bic case if compoenet > 1
    
        #self.gmm_min_comp, self.gmm_min_bic, self.gmm_min_aic, self.gmm_predicted_labels = min_comp, min_bic, min_aic, min_predicted_labels
        ##print("DEBUG: #components, BIC, AIC, ARI, CMProb: ", min_comp, min_bic, min_aic, max_ari, min_predicted_labels, cm_)
        #return min_comp, min_bic, min_aic, max_ari, min_predicted_labels, cm_, var_#sorted(np.sum(cm_, axis=1), reverse=True)[0]


        


if __name__ == '__main__':
    n_component_list = [7]
    n_sample_per_config = 10
    num_sensors = 50
    target_sensors = 35
    tx_power_dBm = np.array([31.0, 33.0, 30.0])
    cmdict = {}
    cmdictp = {}
    vardictp = {}
    sensor_count_data = []
    
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = np.array([
                                                 [100, 100],
                                                 [500,   500],
                                                 [900,   900]
                              ]),
                              d_0_meter=10.0,
                              sigma_sq_db = 5.0, ksensorsloc=[], kprimesensorsloc=[]
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()
    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1
    print('total_distinct_config: ', total_distinct_config)
    n_sample =  n_sample_per_config * total_distinct_config
    sensors = 0
    gsm.generateData(n_sample, False, True, num_sensors)
    sensor_org_data = gsm.interpolateMissingData(pca=num_sensors, x_indices = gsm.training_data_x_indx,
                                                           y_indices =  gsm.training_data_y_indx,
                                                           vals = gsm.training_data_vals,
                                                           labels = gsm.training_data_labels)#would we still interpolate??I think this result should be based on the best
                                                                #algo. discovered from first part - for now it is EMI.
    c, min_bic, min_aic, gmm_min_ari, gmm_predicted_labels, cm, var = gsm.runGMMClustering(n_component_list = (np.array(gsm.cur_training_matrix)).shape[0])
    var_org = gsm.var
    sensor_count_data=[]
###-------------------------------EMSensorSelectionCM---------------------------------####
    for numsensors in range(1, num_sensors):
        
        #gsm.generateData(n_sample, False, True, numsensors)

	gsm.cur_training_matrix = []
    	for mat in sensor_org_data:
        	t = mat[:num_sensors]
        	gsm.cur_training_matrix.append(t)

        c, min_bic, min_aic, gmm_min_ari, gmm_predicted_labels, cm, var = gsm.runGMMClustering(n_component_list = (np.array(gsm.cur_training_matrix)).shape[0])
        print('cm.shape: ', cm)
        cm_all = zip(*cm)
        rowcount = 0
        for row in range(cm.shape[0]):
                proba = max(cm[row,:])
                #print('proba: ', proba)
        	if np.float64(proba) > 0.8:#5/n_component_list.size:
        		rowcount +=1
        #print('row count: ', rowcount)
        if rowcount == n_sample:##can improve logic by using cm[row,:].any()
        	#print('numsensors: ', numsensors)
        	sensors = numsensors
		sensor_count_data = gsm.cur_training_matrix[:,[range(0,sensors)]]
		sensor_count_data = np.squeeze(sensor_count_data, axis=1)
        	break
        	#sensors.append(numsensors)
    print('best k sensors:', numsensors)#sensor_count_data, sensor_org_data, len(sensor_count_data))

    remaining_data_cm = {}    	
    for nminusnprime in range(1, num_sensors-sensors):#target_sensors-int(sensors)):
            gsm.cur_training_matrix = []
            for mat in sensor_org_data:
                t = mat[sensors:sensors+nminusnprime]
                gsm.cur_training_matrix.append(t)
	    print('gsm.cur_training_matrix shape: ', (np.array(gsm.cur_training_matrix)).shape)
            #gsm.cur_training_matrix = np.array(train_matrix)

            #print('cm is: ', cm)
            cm_sum = np.sum(cm, axis=0)
            #print('cm sum: ', cm_sum)
            remaining_data_cm[str(nminusnprime)] = gsm.cur_training_matrix
            cmdictp[str(nminusnprime)] = (sorted(cm_sum))[0]#########check from here
    cmdictp_x = sorted(cmdictp.items(), key=lambda cmdictp: cmdictp[1])
    remaining_sensors_ct = 0
    for keyval in cmdictp_x:
    	print('keyval: ', keyval)
    	if int(keyval[0])>=(target_sensors-sensors):
		remaining_sensors_ct = int(keyval[0])
		break
    print('remaining_sensors_ct: ', remaining_sensors_ct)
    print('target_sensors-sensors: ', target_sensors-sensors)
    sensor_count_remn_data = []
    value = 0
    for k,v in cmdictp_x:
    	if k == str(remaining_sensors_ct):
		value = cmdictp_x.index((k,v))
		print 'value: ', value

    if int(remaining_sensors_ct) >= target_sensors-sensors:	

    	#print('remaining_data_cm.get(cmdictp_x[int(remaining_sensors_ct)]: ' , remaining_data_cm.get(str(cmdictp_x[(remaining_sensors_ct)])))
    	val = remaining_data_cm.get(cmdictp_x[value][0])
	for row in val:
		val1 = row[:target_sensors-sensors]
		sensor_count_remn_data.append(val1)
    gsm.cur_training_matrix = np.hstack( (sensor_count_data,np.array(sensor_count_remn_data))) 
    print('sensor_count_data: ', (sensor_count_data).shape)
    gsm.cur_training_matrix = np.array(gsm.cur_training_matrix)
    print('final training matrix: ', (gsm.cur_training_matrix).shape)
    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels, iem_cm, iem_var = gsm.runGMMClustering(n_component_list = (np.array(gsm.cur_training_matrix)).shape[0])
    print(iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels)
    iem_derived_maps = gsm.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels, x_indices = gsm.training_data_x_indx,
                                               y_indices =  gsm.training_data_y_indx,
                                               vals = gsm.training_data_vals)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = gsm.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)

    print "selected model component# ",iem_min_comp
    print "ARI (GMM vs KMeans): ", iem_min_ari
    print "Average Map Error (GMM vs KMeans): ", iem_avg_map_error


    #---#---------------display the heat maps for GMM IEM-----------------------
    iem_display_mapList = []
    for i in iem_real_map_indices:
        iem_display_mapList.append(gsm.all_maps[i])

    for j in iem_derived_map_indices:
        iem_display_mapList.append( iem_derived_maps[j] )
    ###------------------------------########------------------------------------------###

    gsm.displayMaps(map_list=iem_display_mapList, figFilename = './sensorselection/EMCM_Sensel_iem.png', n_rows=2)


