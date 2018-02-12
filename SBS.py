import numpy as np
import itertools
import pandas as pd
import math
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
import multiprocessing
import matplotlib
matplotlib.use('Agg')

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
                    noise_floor_dB = -60.0
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
        np.random.seed(1009993)
    
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

    def generateDerivedMaps(self, max_x, max_y, predicted_labels, x_indices, y_indices, vals, labels):
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
            print('in generateDerivedMaps', c)

        return derivedMaps


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
            #print('cur_map : ', cur_map)
            #print('i: ', i)
            print('cur_map shape : ', cur_map.shape)
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

    def generateTrainingData(self, n_sample, add_noise, dim_ratio):
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
        appended = False
	maxi = 0       
	x_append=0
	y_append=0
        total_maps = len(self.all_maps)
        map_max_x, map_max_y = self.all_maps[0].shape
        all_loc_indx = list(np.ndindex(map_max_x, map_max_y))
        total_indices = len(all_loc_indx)
	indices_to_be_chosen = max(1, int(np.round(dim_ratio*map_max_x*map_max_y)))#500
	numsensors = int(np.round(indices_to_be_chosen*50.0/100))
        for i in np.arange(n_sample):
            map_indx = np.random.choice(total_maps, 1)[0]
	    indx_to_locs = np.random.choice(total_indices, indices_to_be_chosen, replace = False)
	    chosen_indices = sorted([all_loc_indx[cindx] for cindx in indx_to_locs])
	    x_indx, y_indx = zip(*chosen_indices)
	    #print("number of sensors(positions which will with values): ", p, len(x_indx))
	    #temp_x = (x_indx[0],)
	    #temp_y = (y_indx[0],)
	    temp_x = (np.random.choice(x_indx, 1)[0],)
            temp_y = (np.random.choice(y_indx, 1)[0], )
	    for k in range(len(x_indx)-1):
	    	for l in range(k+1, len(x_indx)):
			if x_indx[l] not in temp_x or y_indx[l] not in temp_y:
				d = (((int(x_indx[l])) - (int(x_indx[k])))**2+((int(y_indx[l]))-(int(y_indx[k])))**2)**0.5
			#print('calculated radius: ', d, x_indx[l], x_indx[k],y_indx[l], y_indx[k])
		    	#print('temp_x: ', str(d)+" >"+str(radius))
				if d > maxi:
					maxi = d
					x_append = x_indx[l]
					y_append = y_indx[l]
					appended = True
		
		if appended and len(temp_x)<numsensors:
			temp_x = temp_x+(x_append,)
			temp_y = temp_y+(y_append,)
			maxi = 0
			appended = False



	    #print('temp_x: ', temp_x)
	    #print('temp_y: ', temp_y)
	    chosen_signal_vals = self.all_maps[map_indx][temp_x , temp_y]
	    #print('chosen_signal_vals: ',chosen_signal_vals, len(chosen_signal_vals))
	    if add_noise:
	    	shadowing_vals = np.random.normal(loc = 0.0,
					      scale = self.sigma_db,
					      size = len(chosen_signal_vals)
					      )
	    	chosen_signal_vals +=  shadowing_vals
	    self.training_data_labels.append(map_indx)
	    self.training_data_vals.append(chosen_signal_vals)
	    #print('number of self.training_data_vals: ', len(self.training_data_vals))
	    self.training_data_x_indx.append(temp_x)
	    self.training_data_y_indx.append(temp_y)

    def prettyPrintSamples(self):
        '''
        mostly for debug
        :return:
        '''
        self.readings={}
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
                #self.readings[(x_indx,y_indx)] = val

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
    
    
    def preprocessTrainingData(self, pca_var_ratio, x_indices, y_indices, vals, labels):
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
            
            for ix, iy in self.index_tuples:
                if (ix, iy) in cur_indices_list:
                    cur_val = cur_vals[cur_indices_list.index((ix, iy))]
                    cur_vector.append(cur_val)
                #else interpolate
                else:
                    interpolated_val = self.idw(xs = x_indx,
                                                ys = y_indx,
                                                vals=cur_vals,
                                                xi = ix,
                                                yi = iy)
                    cur_vector.append(interpolated_val)


            if self.cur_training_matrix is None:
                self.cur_training_matrix = np.array(cur_vector)
            else:
                self.cur_training_matrix = np.vstack( (self.cur_training_matrix,
                                                       np.array(cur_vector))
                                                     )




    def runGMMClusteringForKnownComponents(self, n_components, covariance_type = 'full'):
        '''
            
            :return: ARI value
            '''
                #self.cur_n_component = n_components
        
        gmm = GaussianMixture( n_components = n_components, covariance_type = covariance_type).fit(self.cur_training_matrix)
        predicted_labels = gmm.predict(self.cur_training_matrix)
        probs = gmm.predict_proba(self.cur_training_matrix)
        #print('Cluster Membership probabilities:', probs[:5].round(3))
        cur_ari = adjusted_rand_score(self.training_data_labels, predicted_labels)
        cur_aic = gmm.aic(self.cur_training_matrix)
        cur_bic = gmm.bic(self.cur_training_matrix)
        # print "debug: posterior proba:"
        #print('sorted cm: ', sorted(np.sum(gmm.predict_proba(self.cur_training_matrix), axis=0), reverse=True))
        return cur_bic, cur_aic, cur_ari, predicted_labels
    
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
        #low = 1
        #high = n_component_list
        #while kh > kl:
        #        print('kh , kl: ', kh, kl)
        #        #mid =(kh+kl)/2
        #        cur_bic_kl, cur_aic_kl, cur_ari_kl, predicted_labels_kl = self.runGMMClusteringForKnownComponents(kl,
        #                                                                                covariance_type)


        #        cur_bic_kh, cur_aic_kh, cur_ari_kh, predicted_labels_kh = self.runGMMClusteringForKnownComponents(kh,                                                                                                                                    covariance_type)

        #        if cur_bic_kl < cur_bic_kh: #and cur_bic_kl < min_bic:
        #            kh = (kh + kl)/2
        #            min_comp, min_bic, min_aic, min_ari, min_predicted_labels = kh, \
        #                                                          cur_bic_kh, \
        #                                                          cur_aic_kh, \
        #                                                          cur_ari_kh, \
        #                                                          predicted_labels_kh
        #            kl+=1
        #        elif cur_bic_kh < cur_bic_kl:# and cur_bic_kh < min_bic:                                                                 
        #            kl = (kh + kl)/2
        #            min_comp, min_bic, min_aic, min_ari, min_predicted_labels = kl, \
        #                                                          cur_bic_kl, \
        #                                                          cur_aic_kl, \
        #                                                          cur_ari_kl, \
        #                                                          predicted_labels_kl

        #            kh-=1

        #        print "DEBUG: cur_bic_kl: #components, BIC, AIC, ARI: ",   cur_bic_kl, cur_aic_kl, cur_ari_kl
        #        print "DEBUG: cur_bic_kh: #components, BIC, AIC, ARI: ",   cur_bic_kh, cur_aic_kh, cur_ari_kh
        #        print "DEBUG: min: #components, BIC, AIC, ARI: ",   min_comp, min_bic, min_aic, min_ari
        ##re-run gmm for the min-bic case if compoenet > 1

        #self.gmm_min_comp, self.gmm_min_bic, self.gmm_min_aic, self.gmm_predicted_labels = \
        #         min_comp,          min_bic,          min_aic,      min_predicted_labels
	min_comp = kh
        min_bic, min_aic, min_ari, min_predicted_labels = self.runGMMClusteringForKnownComponents(kh,                                                                                                                                    covariance_type)

        return min_comp, min_bic, min_aic, min_ari, min_predicted_labels  

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
        print('min_err_row_indx, min_err_col_indx : ' , min_err_row_indx, min_err_col_indx )
        #if dimensions are unequal, re-run association
        new_min_error_row_indx = []
        new_min_error_col_indx = []
        if len(maps_A_dBm) > len(maps_B_dBm): #unassociated rows there
            print('len(maps_A_dBm) > len(maps_B_dBm)')
            unassociated_rows = list(  set(range( len(maps_A_dBm) )) - set(min_err_row_indx)  )
            new_error_matrix = error_matrix[ unassociated_rows , :]
            _,  new_min_error_col_indx = linear_sum_assignment(new_error_matrix)
            new_min_error_row_indx = unassociated_rows
        elif len(maps_A_dBm) < len(maps_B_dBm): #unassociated cols there
            print('len(maps_A_dBm) < len(maps_B_dBm)')
            unassociated_cols = list(  set(range( len(maps_B_dBm) ) ) - set(min_err_col_indx)  )
            new_error_matrix = error_matrix[ : , unassociated_cols].T
            _, new_min_error_row_indx = linear_sum_assignment(new_error_matrix)
            new_min_error_col_indx = unassociated_cols
        print('new_min_error_col_indx: ', new_min_error_col_indx)
        print('new_min_error_row_indx: ', new_min_error_row_indx)
        map_A_assoc_indices = list(min_err_row_indx)+list(new_min_error_row_indx)
        map_B_assoc_indices = list(min_err_col_indx) + list(new_min_error_col_indx)
        print('min_err_row_indx: ', min_err_row_indx)
        print('new_min_error_col_indx: ', new_min_error_col_indx)
	print('new_min_error_col_indx: ', new_min_error_col_indx)
        print('new_min_error_row_indx: ', new_min_error_row_indx)
        print('min_err_col_indx: ', min_err_col_indx)
        print('np.ix_(map_A_assoc_indices, map_B_assoc_indices):', np.ix_(map_A_assoc_indices, map_B_assoc_indices))
        avg_map_error =  np.average(  error_matrix[np.ix_(map_A_assoc_indices, map_B_assoc_indices)] )
        return map_A_assoc_indices, map_B_assoc_indices, avg_map_error

                                                                       

    def findCovBasedSensors(self):
        '''
        
        :return: ARI value
        '''


        #temp = self.cur_training_matrix
        #print('self.cur_training_matrix:', self.cur_training_matrix.shape)
        #pca = PCA(pca_var_ratio, whiten=True)
        #self.cur_training_matrix =  pca.fit_transform(self.cur_training_matrix)
        
        #print("Feature Reduction for ",pca_var_ratio," variances: ", pca.n_components_ ," / ",len(self.index_tuples), self.cur_training_matrix.shape, temp.shape)
        #print self.cur_training_matrix.shape
        #print self.cur_training_matrix
        #self.cur_n_component = n_components
        #probs = self.predict_proba(self.cur_training_matrix)
        #print('Cluster Membership probabilities:', np.sum(gmm.predict_proba(self.cur_training_matrix), axis=1))
        #print('variance: ', np.sum(pca.explained_variance_ratio_))
        #print('Cluster Membership probabilities:', probs[:5].round(3))
        #return np.sum(pca.explained_variance_ratio_)
        # print "debug: posterior proba:"
        # print np.sum(self.cur_gmm.predict_proba(self.cur_training_matrix), axis=1)



#---------------------Display Maps--------------------------------------------------#
#--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])


def iteration1():
####3rd iteration-------------###
    file = open("sbs_5.txt","w")
    pca_var_ratio = 0.9999999999
    n_component_list = 31
    #n_component_list = np.arange(1, 10)
    n_sample_per_config = 50
    tx_power_dBm = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    max_x_meter = 100.0
    max_y_meter = 100.0
   
    gsm = GenerateSpectrumMap(max_x_meter = max_x_meter,
                              max_y_meter = max_y_meter,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = np.array([
                                                [5, 5],
                                                [100, 100],
                                                [50, 50],
                                                [75, 75],
                                                [25, 25]
    
                              ]),
                              d_0_meter=5.0,
                              sigma_sq_db = 10.0
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()
    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1
    print('total_distinct_config: ', total_distinct_config)
    n_sample =  n_sample_per_config * total_distinct_config
    gsm.generateTrainingData(n_sample, True, 5.0/100)
    gsm.preprocessTrainingData(pca_var_ratio=pca_var_ratio, x_indices = gsm.training_data_x_indx,
                               y_indices =  gsm.training_data_y_indx,
                               vals = gsm.training_data_vals,
                               labels = gsm.training_data_labels)

    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = gsm.runGMMClustering(n_component_list = n_component_list)#(np.array(gsm.cur_training_matrix)).shape[0])

    iem_derived_maps = gsm.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels, x_indices = gsm.training_data_x_indx,
                                               y_indices =  gsm.training_data_y_indx,
                                               vals = gsm.training_data_vals,
                                               labels = gsm.training_data_labels)
    print 'iem_derived_maps: ', len(iem_derived_maps)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = gsm.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)
    print "selected model component# ",iem_min_comp
    print "ARI: ", iem_min_ari
    print "Average Map Error: ", iem_avg_map_error

    print 'iem_derived_map_indices: ', iem_derived_map_indices
    print "selected model components: ", iem_min_comp
    print "ARI : ", iem_min_ari
    print "Average Map Error: ", iem_avg_map_error
    file.write(str(5)+","+str(iem_min_ari)+ ","+str(iem_avg_map_error))
    file.close()


#---------------------Display Maps--------------------------------------------------#
#--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])



def iteration2():
####3rd iteration-------------###
    file = open("sbs_6.txt","w")
    pca_var_ratio = 0.9999999999
    n_component_list = 31
    #n_component_list = np.arange(1, 10)
    n_sample_per_config = 50
    tx_power_dBm = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    max_x_meter = 100.0
    max_y_meter = 100.0

    gsm = GenerateSpectrumMap(max_x_meter = max_x_meter,
                              max_y_meter = max_y_meter,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = np.array([
                                                [5, 5],
                                                [100, 100],
                                                [50, 50],
                                                [75, 75],
                                                [25, 25]
    
                              ]),
                              d_0_meter=5.0,
                              sigma_sq_db = 10.0
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()
    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1
    print('total_distinct_config: ', total_distinct_config)
    n_sample =  n_sample_per_config * total_distinct_config
    gsm.generateTrainingData(n_sample, True, 6.0/100)
    gsm.preprocessTrainingData(pca_var_ratio=pca_var_ratio, x_indices = gsm.training_data_x_indx,
                               y_indices =  gsm.training_data_y_indx,
                               vals = gsm.training_data_vals,
                               labels = gsm.training_data_labels)

    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = gsm.runGMMClustering(n_component_list = n_component_list)#(np.array(gsm.cur_training_matrix)).shape[0])

    iem_derived_maps = gsm.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels, x_indices = gsm.training_data_x_indx,
                                               y_indices =  gsm.training_data_y_indx,
                                               vals = gsm.training_data_vals,
                                               labels = gsm.training_data_labels)
    print 'iem_derived_maps: ', len(iem_derived_maps)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = gsm.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)
    print "selected model component# ",iem_min_comp
    print "ARI: ", iem_min_ari
    print "Average Map Error: ", iem_avg_map_error

    print 'iem_derived_map_indices: ', iem_derived_map_indices
    print "selected model components: ", iem_min_comp
    print "ARI : ", iem_min_ari
    print "Average Map Error: ", iem_avg_map_error
    file.write(str(6)+","+str(iem_min_ari)+ ","+str(iem_avg_map_error))
    file.close()



def iteration3():
####3rd iteration-------------###
    file = open("sbs_7.txt","w")
    pca_var_ratio = 0.9999999999
    n_component_list = 31
    #n_component_list = np.arange(1, 10)
    n_sample_per_config = 50
    tx_power_dBm = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    max_x_meter = 100.0
    max_y_meter = 100.0

    gsm = GenerateSpectrumMap(max_x_meter = max_x_meter,
                              max_y_meter = max_y_meter,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = np.array([
                                                [5, 5],
                                                [100, 100],
                                                [50, 50],
                                                [75, 75],
                                                [25, 25]
    
                              ]),
                              d_0_meter=5.0,
                              sigma_sq_db = 10.0
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()
    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1
    print('total_distinct_config: ', total_distinct_config)
    n_sample =  n_sample_per_config * total_distinct_config
    gsm.generateTrainingData(n_sample, True, 7.0/100)
    gsm.preprocessTrainingData(pca_var_ratio=pca_var_ratio, x_indices = gsm.training_data_x_indx,
                               y_indices =  gsm.training_data_y_indx,
                               vals = gsm.training_data_vals,
                               labels = gsm.training_data_labels)

    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = gsm.runGMMClustering(n_component_list = n_component_list)#(np.array(gsm.cur_training_matrix)).shape[0])

    iem_derived_maps = gsm.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels, x_indices = gsm.training_data_x_indx,
                                               y_indices =  gsm.training_data_y_indx,
                                               vals = gsm.training_data_vals,
                                               labels = gsm.training_data_labels)
    print 'iem_derived_maps: ', len(iem_derived_maps)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = gsm.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)

    print 'iem_derived_map_indices: ', iem_derived_map_indices
    print "selected model components: ", iem_min_comp
    print "ARI : ", iem_min_ari
    print "Average Map Error: ", iem_avg_map_error
    file.write(str(7)+","+str(iem_min_ari)+ ","+str(iem_avg_map_error))
    file.close()



def iteration4():
####3rd iteration-------------###
    file = open("sbs_8.txt","w")
    pca_var_ratio = 0.9999999999
    n_component_list = 31
    #n_component_list = np.arange(1, 10)
    n_sample_per_config = 50
    tx_power_dBm = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    max_x_meter = 100.0
    max_y_meter = 100.0

    gsm = GenerateSpectrumMap(max_x_meter = max_x_meter,
                              max_y_meter = max_y_meter,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = np.array([
                                                [5, 5],
                                                [100, 100],
                                                [50, 50],
                                                [75, 75],
                                                [25, 25]
    
                              ]),
                              d_0_meter=5.0,
                              sigma_sq_db = 10.0
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()
    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1
    print('total_distinct_config: ', total_distinct_config)
    n_sample =  n_sample_per_config * total_distinct_config
    gsm.generateTrainingData(n_sample, True, 8.0/100)
    gsm.preprocessTrainingData(pca_var_ratio=pca_var_ratio, x_indices = gsm.training_data_x_indx,
                               y_indices =  gsm.training_data_y_indx,
                               vals = gsm.training_data_vals,
                               labels = gsm.training_data_labels)

    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = gsm.runGMMClustering(n_component_list = n_component_list)#(np.array(gsm.cur_training_matrix)).shape[0])

    iem_derived_maps = gsm.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels, x_indices = gsm.training_data_x_indx,
                                               y_indices =  gsm.training_data_y_indx,
                                               vals = gsm.training_data_vals,
                                               labels = gsm.training_data_labels)
    print 'iem_derived_maps: ', len(iem_derived_maps)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = gsm.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)
    print "selected model component# ",iem_min_comp
    print "ARI: ", iem_min_ari
    print "Average Map Error: ", iem_avg_map_error

    print 'iem_derived_map_indices: ', iem_derived_map_indices
    file.write(str(8)+","+str(iem_min_ari)+ ","+str(iem_avg_map_error))
    file.close()






#if __name__ == '__main__':
#    pca_var_ratio = 0.9999999999
#    n_component_list = 31
#    #n_component_list = np.arange(1, 10)
#    n_sample_per_config = 50
#    tx_power_dBm = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
#    max_x_meter = 100.0
#    max_y_meter = 100.0
#    
#    gsm = GenerateSpectrumMap(max_x_meter = max_x_meter,
#                              max_y_meter = max_y_meter,
#                              tx_power_dBm = tx_power_dBm,
#                              tx_loc = np.array([
#                                                [5, 5],
#                                                [100, 100],
#                                                [50, 50],
#                                                [75, 75],
#                                                [25, 25]  
#                                                 
#                              ]),
#                              d_0_meter=5.0,
#                              sigma_sq_db = 10.0
#                              )
#    gsm.generateIndividualMap()
#    gsm.generateAllCombinationMap()
#    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1
#    print('total_distinct_config: ', total_distinct_config)
#    n_sample =  n_sample_per_config * total_distinct_config
#    gsm.generateTrainingData(n_sample, True, 10.0/100)
#    gsm.preprocessTrainingData(pca_var_ratio=pca_var_ratio, x_indices = gsm.training_data_x_indx,
#                               y_indices =  gsm.training_data_y_indx,
#                               vals = gsm.training_data_vals,
#                               labels = gsm.training_data_labels)
#
#    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = gsm.runGMMClustering(n_component_list = n_component_list)#(np.array(gsm.cur_training_matrix)).shape[0])
#
#    iem_derived_maps = gsm.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
#                                               max_y = gsm.get_map_grid_y(),
#                                               predicted_labels=iem_predicted_labels, x_indices = gsm.training_data_x_indx,
#                                               y_indices =  gsm.training_data_y_indx,
#                                               vals = gsm.training_data_vals,
#                                               labels = gsm.training_data_labels)
#    print 'iem_derived_maps: ', len(iem_derived_maps)
#    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = gsm.avgPairwiseMapError(gsm.all_maps,
#                                                                                              iem_derived_maps)
#    print "selected model component# ",iem_min_comp
#    print "ARI: ", iem_min_ari
#    print "Average Map Error: ", iem_avg_map_error											      
#
#    print 'iem_derived_map_indices: ', iem_derived_map_indices
#
#    #---#---------------display the heat maps for GMM IEM-----------------------
#    iem_display_mapList = []
#    for i in iem_real_map_indices:
#        iem_display_mapList.append(gsm.all_maps[i])
#
#    for j in iem_derived_map_indices:
#        iem_display_mapList.append( iem_derived_maps[j] )
#
#    gsm.displayMaps(map_list=iem_display_mapList, figFilename = './SBS_Sensel_iem.png', n_rows=2)





if __name__ == '__main__':
    p1 = multiprocessing.Process(target=iteration1)
    p2 = multiprocessing.Process(target=iteration2)
    p3 = multiprocessing.Process(target=iteration3)
    p4 = multiprocessing.Process(target=iteration4)

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    print("Done!")

