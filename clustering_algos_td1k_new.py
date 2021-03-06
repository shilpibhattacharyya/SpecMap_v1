import numpy as np
import itertools
from sklearn.mixture import GaussianMixture
from data_generator import GenerateSpectrumMap
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
np.set_printoptions(threshold=np.inf)
from sklearn.metrics import confusion_matrix
import multiprocessing

class SpecMapClustering:
    '''
    tasks:
    i) read the training data
    ii) do preprocessing(interpolation etc.) on the training data
    iii) declare model and fit the data
    iv) find the predicted labels and compute accuracy
    '''
    def __init__(self, x_indices, y_indices, vals, labels):
        '''

        '''
        self.x_indices = x_indices
        self.y_indices = y_indices
        self.vals = vals
        self.labels = labels
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
            #if dist_sq <=0.0:
            #    dist_sq = 0.0000001
            w = 1.0/dist_sq
            sum_of_weights += w
            sum_of_weight_times_val += z*w
        interp_val = sum_of_weight_times_val/sum_of_weights
        return interp_val

    def rec_range(self, n):
 	"""Takes a natural number n and returns a tuple of numbers starting with 0 and ending before n.

    	Natural Number -> tuple"""
    	if n == 0:
        	return 
    	elif n == 1:
        	return (0,)
    	else:
        	return self.rec_range(n-1) + (n-1,)


    def preprocessTrainingData(self, pca_var_ratio = 1.0):
        '''
        handle missing values by interpolation
        :return:
        '''
        self.cur_training_matrix = None
        #find set of all unique features as tuple-list of coordinates
        self.index_tuples = []
        for x_indx, y_indx in zip(self.x_indices, self.y_indices):
            self.index_tuples.extend(zip(x_indx, y_indx))
        self.index_tuples = sorted(list(set(self.index_tuples)))
        #now create a tr:aining matrix for each datapointi
	self.locandsensorvals = []
        for x_indx, y_indx, cur_vals, label in zip( self.x_indices,
                                                self.y_indices,
                                                self.vals,
                                                self.labels
                                               ):
	    cur_vector = []
            #scan for all features, if the value exists, copy, if not interpolate
            cur_indices = zip(x_indx, y_indx)
	    locwithsensorval = {}
            for ix, iy in self.index_tuples:###check this logic
                if (ix, iy) in cur_indices:
                    cur_val = cur_vals[ cur_indices.index((ix, iy))]
                    cur_vector.append(cur_val)
		    locwithsensorval[str((ix,iy))]=cur_val
                #else interpolate
                else:
		    #print('interploation for xs and ys: ', x_indx, y_indx)
                    interplated_val = self.idw(xs = x_indx,
                                               ys = y_indx,
                                               vals=cur_vals,
                                               xi = ix,
                                               yi = iy)
                    cur_vector.append(interplated_val)
            if self.cur_training_matrix is None:
                self.cur_training_matrix = np.array(cur_vector)
            else:
                self.cur_training_matrix = np.vstack( (self.cur_training_matrix,#it has much lesser values than original grid even after interpoltaion
                                                       np.array(cur_vector))
                                                     )
	    self.locandsensorvals.append(locwithsensorval)
	self.cur_training_matrix1 = self.cur_training_matrix
        if pca_var_ratio < 1.0:
            pca = PCA(pca_var_ratio)
            self.cur_training_matrix =  pca.fit_transform(self.cur_training_matrix)
            print "Reduced components: ", pca.n_components_ ," / ",len(self.index_tuples)
  

    def preprocessTrainingDataEMII(self, pca_var_ratio=1.0):
        ''' 
        handle missing values by interpolation
        :return:
        '''
        self.cur_training_matrix = None
        #find set of all unique features as tuple-list of coordinates
	self.index_tuples = []
        #print(zip(self.x_indices, self.y_indices))
        for x_indx, y_indx in zip(self.x_indices, self.y_indices):
            self.index_tuples.extend(zip(x_indx, y_indx))
        self.index_tuples = sorted(list(set(self.index_tuples)))
        #now create a tr:aining matrix for each datapoint
	self.locandsensorvals = []
        for x_indx, y_indx, cur_vals, label in zip( self.x_indices,
                                                self.y_indices,
                                                self.vals,
                                                self.labels
                                               ):  
            cur_vector = []
            #scan for all features, if the value exists, copy, if not interpolate
            cur_indices = zip(x_indx, y_indx)
	    locwithsensorval = {}
            for ix, iy in self.index_tuples:###check this logic
                if (ix, iy) in cur_indices:
                    cur_val = cur_vals[ cur_indices.index((ix, iy))]
                    cur_vector.append(cur_val)
		    locwithsensorval[str((ix,iy))]=cur_val
                #else interpolate
		else:
                    interplated_val = self.idw(xs = x_indx,
                                               ys = y_indx,
                                               vals=cur_vals,
                                               xi = ix,
                                               yi = iy)
                    cur_vector.append(interplated_val)
            if self.cur_training_matrix is None:
                self.cur_training_matrix = np.array(cur_vector)
            else:
                self.cur_training_matrix = np.vstack( (self.cur_training_matrix, np.array(cur_vector)))
	    self.locandsensorvals.append(locwithsensorval)
	self.cur_training_matrix1 = self.cur_training_matrix
	if pca_var_ratio < 1.0: 
            pca = PCA(pca_var_ratio)
            self.cur_training_matrix =  pca.fit_transform(self.cur_training_matrix)
	    print "Reduced components: ", pca.n_components_ ," / ",len(self.index_tuples)


    def runKMeansClusteringForKnownComponents(self, n_components = 2):
        '''

        :param n_components:
        :return:
        '''
        self.kms = KMeans(n_clusters= n_components).fit(self.cur_training_matrix)
     	self.kms_predicted_labels = self.kms.predict(self.cur_training_matrix)
        self.kms_ari = adjusted_rand_score( self.labels, self.kms_predicted_labels)
        return self.kms_predicted_labels, self.kms_ari

    def runGMMClusteringForKnownComponents(self, n_components, coverience_type):
        '''

        :return: ARI value
        '''
        gmm = GaussianMixture( n_components = n_components,
                               covariance_type = coverience_type).fit(self.cur_training_matrix)
	print 'no. of iterations taken to converge: ', gmm.n_iter_
	print 'score: ', gmm.score(self.cur_training_matrix), 'log likelihood: ', gmm.lower_bound_

	#data_new = PCA(0.95).inverse_transform(data)
        predicted_labels = gmm.predict(self.cur_training_matrix)

        cur_ari = adjusted_rand_score(self.labels, predicted_labels)
        cur_aic = gmm.aic(self.cur_training_matrix)
        cur_bic = gmm.bic(self.cur_training_matrix)
        # print np.sum(self.cur_gmm.predict_proba(self.cur_training_matrix), axis=1)
        return cur_bic, cur_aic, cur_ari, predicted_labels

    def runGMMClustering(self, n_component_list, coverience_type = 'full'):
        '''
        based on bic find the best # of components
        :param coverience_type:
        :return:
        '''
        min_comp, min_bic, min_aic, min_ari, min_predicted_labels = None, \
                                                                float('inf'), \
                                                                float('inf'), \
                                                                float('inf'), \
                                                                []

        kl =  1
        kh =  n_component_list[0]
        #low = 1
        #high = n_component_list

        #while kh > kl:
        #	#print('kh , kl: ', kh, kl)
        #	#mid =(kh+kl)/2
        #	cur_bic_kl, cur_aic_kl, cur_ari_kl, predicted_labels_kl = self.runGMMClusteringForKnownComponents(kl,
        #										coverience_type)


        #	cur_bic_kh, cur_aic_kh, cur_ari_kh, predicted_labels_kh = self.runGMMClusteringForKnownComponents(kh,																	 coverience_type)

        #	if cur_bic_kl < cur_bic_kh: #and cur_bic_kl < min_bic:
        #	    kh = (kh + kl)/2
        #	    min_comp, min_bic, min_aic, min_ari, min_predicted_labels = kh, \
        #                                                          cur_bic_kh, \
        #                                                          cur_aic_kh, \
        #                                                          cur_ari_kh, \
        #                                                          predicted_labels_kh
	#	    kl+=1
        #	elif cur_bic_kh < cur_bic_kl:# and cur_bic_kh < min_bic: 								 
        #	    kl = (kh + kl)/2
        #      	    min_comp, min_bic, min_aic, min_ari, min_predicted_labels = kl, \
        #                                                          cur_bic_kl, \
        #                                                          cur_aic_kl, \
        #                                                          cur_ari_kl, \
        #                                                          predicted_labels_kl

        #	    kh-=1

        #	#print "DEBUG: cur_bic_kl: #components, BIC, AIC, ARI: ",   cur_bic_kl, cur_aic_kl, cur_ari_kl
        #	#print "DEBUG: cur_bic_kh: #components, BIC, AIC, ARI: ",   cur_bic_kh, cur_aic_kh, cur_ari_kh
        #        #print "DEBUG: min: #components, BIC, AIC, ARI: ",   min_comp, min_bic, min_aic, min_ari
        ##re-run gmm for the min-bic case if compoenet > 1

        #self.gmm_min_comp, self.gmm_min_bic, self.gmm_min_aic, self.gmm_predicted_labels = \
        #         min_comp,          min_bic,          min_aic,      min_predicted_labels
	min_comp = n_component_list[0]
	min_bic, min_aic, min_ari, min_predicted_labels =  self.runGMMClusteringForKnownComponents(kh, coverience_type)

        return min_comp, min_bic, min_aic, min_ari, min_predicted_labels



    def generateDerivedMaps(self, max_x, max_y, predicted_labels):
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
                    for x_indx, y_indx, val in zip(self.x_indices[k],
                                                   self.y_indices[k],
                                                   self.vals[k]):
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

        #mapA_mW[mapA_dBm == 0.0] = 0.000001 #to avoid div-by-zero

        error_A_B = (np.abs(mapB_mW - mapA_mW)) / mapA_mW
        avg_error = np.average(error_A_B)
        return avg_error*100

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

        if len(maps_A_dBm) > len(maps_B_dBm): #unassociated rows there
            unassociated_rows = np.array([x for x in range(len(maps_A_dBm)) if x not in min_err_row_indx])
            while unassociated_rows.size >0:
                new_error_matrix = error_matrix[ unassociated_rows , :]
                new_min_error_row_indx,  new_min_error_col_indx = linear_sum_assignment(new_error_matrix)
                new_min_error_row_indx = unassociated_rows[ new_min_error_row_indx ]
                min_err_row_indx = np.concatenate( (min_err_row_indx, new_min_error_row_indx ) )
                min_err_col_indx = np.concatenate( ( min_err_col_indx, new_min_error_col_indx ) )
                unassociated_rows = np.array( [x for x in unassociated_rows if x not in new_min_error_row_indx ] )
        elif len(maps_A_dBm) < len(maps_B_dBm): #unassociated cols there
            unassociated_cols = np.array([x for x in range(len(maps_B_dBm)) if x not in min_err_col_indx])
            while unassociated_cols.size>0:
                new_error_matrix = error_matrix[ : , unassociated_cols]
                new_min_error_row_indx,  new_min_error_col_indx = linear_sum_assignment(new_error_matrix)
                new_min_error_col_indx = unassociated_cols[ new_min_error_col_indx ]
                min_err_row_indx = np.concatenate( (min_err_row_indx, new_min_error_row_indx ) )
                min_err_col_indx = np.concatenate( (min_err_col_indx, new_min_error_col_indx ) )
                unassociated_cols = np.array( [x for x in unassociated_cols if x not in new_min_error_col_indx ] )
        print 'min_err_row_indx: ', min_err_row_indx
        #print 'new_min_error_row_indx: ', new_min_error_row_indx
        print 'min_err_col_indx: ', min_err_col_indx
        #print 'new_min_error_col_indx: ', new_min_error_col_indx
        avg_map_error =  np.average(  error_matrix[min_err_row_indx, min_err_col_indx] )
        return min_err_row_indx, min_err_col_indx , avg_map_error

#    def avgPairwiseMapError(self,maps_A_dBm, maps_B_dBm):
#        '''
#        compute bipartite graph between maps_A_dBm, and maps_B_dBm based on minimum error
#        and generate avg error
#        :param maps_A_dBm:
#        :param maps_B_dBm:
#        :return: mapA indices, corresponding mapB indices, avg_map_error
#        '''
#        error_matrix = np.zeros( shape = ( len(maps_A_dBm), len(maps_B_dBm) ) )
#        for i,cur_map_A in enumerate(maps_A_dBm):
#            for j, cur_map_B in enumerate(maps_B_dBm):
#                error_matrix[i][j] = self.computeMapAccuracy(cur_map_A, cur_map_B)
#        #if dimensions are equal, just associate row-wise mins
#        min_err_row_indx, min_err_col_indx = linear_sum_assignment(error_matrix)
#	#print('min_err_row_indx, min_err_col_indx : ' , min_err_row_indx, min_err_col_indx )
#        #if dimensions are unequal, re-run association
#        new_min_error_row_indx = []
#        new_min_error_col_indx = []
#        if len(maps_A_dBm) > len(maps_B_dBm): #unassociated rows there
#	    #print('len(maps_A_dBm) > len(maps_B_dBm)')
#            unassociated_rows = list(  set(range( len(maps_A_dBm) )) - set(min_err_row_indx)  )
#            new_error_matrix = error_matrix[ unassociated_rows , :]
#            _,  new_min_error_col_indx = linear_sum_assignment(new_error_matrix)
#            new_min_error_row_indx = unassociated_rows
#        elif len(maps_A_dBm) < len(maps_B_dBm): #unassociated cols there
#	    #print('len(maps_A_dBm) < len(maps_B_dBm)')
#            unassociated_cols = list(  set(range( len(maps_B_dBm) ) ) - set(min_err_col_indx)  )
#            new_error_matrix = error_matrix[ : , unassociated_cols].T
#            _, new_min_error_row_indx = linear_sum_assignment(new_error_matrix)
#            new_min_error_col_indx = unassociated_cols
#	#print('new_min_error_col_indx: ', new_min_error_col_indx)
#	#print('new_min_error_row_indx: ', new_min_error_row_indx)
#
#        map_A_assoc_indices = list(min_err_row_indx)+list(new_min_error_row_indx)
#        map_B_assoc_indices = list(min_err_col_indx) + list(new_min_error_col_indx)
#	#print('min_err_row_indx: ', min_err_row_indx)
#	#print('new_min_error_col_indx: ', new_min_error_col_indx)
#	#print('new_min_error_row_indx: ', new_min_error_row_indx)
#	#print('min_err_col_indx: ', min_err_col_indx)
#	#print('np.ix_(map_A_assoc_indices, map_B_assoc_indices):', np.ix_(map_A_assoc_indices, map_B_assoc_indices))
#        avg_map_error =  np.average(  error_matrix[np.ix_(map_A_assoc_indices, map_B_assoc_indices)] )
#        return map_A_assoc_indices, map_B_assoc_indices, avg_map_error
#

###-----------------------------------------------------------------------------------------------------------------------------------
    def runIIGMMClusteringForKnownComponents(self, n_components=7,
                                             coverience_type ='full',
                                             max_iteration=1):
        '''

        :param n_components:
        :param coverience_type:
        :return:
        '''
        labels = []
        temp_m = None#means_init
        temp_p = None#precisions_init
        temp_w = None#weights_init
        temp_c = 0.0#covariances_
        temp_ll = 1.0 #log likelihod of best fit of EM
        self.prev_labels = self.labels

        for loop in range(1, max_iteration+1):

                self.cur_gmm = GaussianMixture( n_components = n_components,
                                        covariance_type = coverience_type,#means_init = temp_m,
                                        #precisions_init = temp_p, weights_init = temp_w ,
                                        warm_start=True, #verbose=3,
                                        max_iter=1
                                        ).fit(self.cur_training_matrix)
                means_init = self.cur_gmm.means_
                precisions_init = self.cur_gmm.precisions_
                weights_init = self.cur_gmm.weights_
                covariances_ = self.cur_gmm.covariances_
                log_likelihood = self.cur_gmm.score(self.cur_training_matrix)#self.cur_gmm.lower_bound_
                print 'log_likelihood: ', log_likelihood, 'score: ', self.cur_gmm.score(self.cur_training_matrix)
                print 'log likelihood percent gain: ', np.abs((log_likelihood - temp_ll)/temp_ll)

                #if np.sum(np.subtract(means_init, temp_m))< 0.5 and np.sum(np.subtract(covariances_,temp_c))<0.25:
                #        #print 'converged'
                #        break
		#if log_likelihood  < 0.049: #and np.sum(np.subtract(covariances_,temp_c))<0.25:
                #        print 'converged'
                #        break

                if (np.abs((log_likelihood - temp_ll)/temp_ll)) < 0.049: #and np.sum(np.subtract(covariances_,temp_c))<0.25:
                        print 'converged'
			#labels = self.cur_gmm.predict(self.cur_training_matrix)
                        break

                temp_m = means_init
                temp_p = precisions_init
                temp_w = weights_init
                temp_c = covariances_
                temp_ll = log_likelihood

                #print('converged_: ', self.cur_gmm.converged_)
                cm = self.cur_gmm.predict_proba(self.cur_training_matrix)
		print 'cm data type: ', cm.dtype
                labels = self.cur_gmm.predict(self.cur_training_matrix)
		#print 'len albels: ', len(labels), labels



				
                for n in range(len(labels)):
                        for (i,j) in self.index_tuples:
                                if str((i,j)) not in self.locandsensorvals[n]:
					#print 'self.cur_training_matrix[n][indx]: ', self.cur_training_matrix[n][indx]
                                        indx = self.index_tuples.index((i,j))
                                        #temp = self.cur_training_matrix[n][indx]
                                        label= labels[n]
                                        simlabels = getallsimilarlabels(label, labels)
                                        t=0.0
                                        weight=0.0
                                        for k in simlabels:
                                                t= t + self.cur_training_matrix1[k][indx]*cm[k][label]
						#print 'cm[k][label]: ', cm[k][label]
                                                weight+=cm[k][label]
                                self.cur_training_matrix1[n][indx] = t/weight
				#print 'self.cur_training_matrix[n][indx] out: ', self.cur_training_matrix[n][indx]
		pca = PCA(0.90)
               	self.cur_training_matrix = pca.fit_transform(self.cur_training_matrix1)
		print 'reduced components: ', pca.n_components_
        self.gmm_min_predicted_labels = self.cur_gmm.predict(self.cur_training_matrix)#self.cur_gmm.predict(self.cur_training_matrix)

	cur_ari = adjusted_rand_score(self.labels, self.gmm_min_predicted_labels)
	cur_aic = self.cur_gmm.aic(self.cur_training_matrix)
	cur_bic = self.cur_gmm.bic(self.cur_training_matrix)
                #self.prev_labels =  self.gmm_min_predicted_labels
        return  cur_bic, cur_aic, cur_ari, self.gmm_min_predicted_labels

#    def runIIGMMClusteringForKnownComponents(self, n_components=7,
#                                             coverience_type ='full',
#                                             max_iteration=1):
#        '''
#
#        :param n_components:
#        :param coverience_type:
#        :return:
#        '''
#	labels = []	
#	temp_m = 0.0#means_init
#        temp_p = 0.0#precisions_init
#        temp_w = 0.0#weights_init
#        temp_c = 0.0#covariances_
#		
#	for loop in range(max_iteration):
#
#		self.cur_gmm = GaussianMixture( n_components = n_components,
#                                        covariance_type = coverience_type,#means_init = means_init,
#					#precisions_init = precisions_init, weights_init = weights_init,
#                                        warm_start=False, #verbose=3,
#                                        max_iter=1
#                                        ).fit(self.cur_training_matrix)
#		means_init = self.cur_gmm.means_
#        	precisions_init = self.cur_gmm.precisions_
#		weights_init = self.cur_gmm.weights_
#		covariances_ = self.cur_gmm.covariances_
#
#                if np.sum(np.subtract(means_init, temp_m))< 0.5 and np.sum(np.subtract(covariances_,temp_c))<0.25:
#                        #print 'converged'
#                        break
#
#
#		temp_m = means_init
#                temp_p = precisions_init
#                temp_w = weights_init
#                temp_c = covariances_
#
#		#print('converged_: ', self.cur_gmm.converged_)
#		cm = self.cur_gmm.predict_proba(self.cur_training_matrix)
#		labels = self.cur_gmm.predict(self.cur_training_matrix)
#
#		for n in range(len(labels)):
#			for (i,j) in self.index_tuples:
#				if str((i,j)) not in self.locandsensorvals[n]:
#					indx = self.index_tuples.index((i,j))
#					temp = self.cur_training_matrix[n][indx]
#					label= labels[n]
#					simlabels = getallsimilarlabels(label, labels)
#					t=0.0
#					weight=0.0
#					for k in simlabels:
#						t= t + self.cur_training_matrix[k][indx]*cm[k][label]
#						weight+=cm[k][label]
#					self.cur_training_matrix[n][indx] = t/weight
#
#		self.gmm_min_predicted_labels = labels#self.cur_gmm.predict(self.cur_training_matrix)
#		cur_ari = adjusted_rand_score(self.labels, self.gmm_min_predicted_labels)
#		cur_aic = self.cur_gmm.aic(self.cur_training_matrix)
#		cur_bic = self.cur_gmm.bic(self.cur_training_matrix)
#	return  cur_bic, cur_aic, cur_ari, labels




    def runGMMClusteringII(self, n_component_list, coverience_type = 'full'):
        ''' 
        based on bic find the best # of components
        :param coverience_type:
        :return:
        '''
        min_comp, min_bic, min_aic, min_ari, min_predicted_labels = None, \
                                                                float('inf'), \
                                                                float('inf'), \
                                                                float('inf'), \
                                                                []  

        kl =  1
        kh =  n_component_list[0]#n_component_list
        #low = 1 
        #high = n_component_list

        #while kh > kl: 
        #        #print('kh , kl: ', kh, kl) 
        #        #mid =(kh+kl)/2
        #        cur_bic_kl, cur_aic_kl, cur_ari_kl, predicted_labels_kl = self.runIIGMMClusteringForKnownComponents(kl,
        #                                                                                coverience_type, 50)


        #        cur_bic_kh, cur_aic_kh, cur_ari_kh, predicted_labels_kh = self.runIIGMMClusteringForKnownComponents(kh,                                                                                                                                    coverience_type, 50)
        #            
        #	if cur_bic_kl < cur_bic_kh:
        #	    kh = (kh + kl)/2
        #           min_comp, min_bic, min_aic, min_ari, min_predicted_labels = kh, \
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

        #        #print "DEBUG: cur_bic_kl: #components, BIC, AIC, ARI: ",   cur_bic_kl, cur_aic_kl, cur_ari_kl
        #        #print "DEBUG: cur_bic_kh: #components, BIC, AIC, ARI: ",   cur_bic_kh, cur_aic_kh, cur_ari_kh
        #        #print "DEBUG: min: #components, BIC, AIC, ARI: ",   min_comp, min_bic, min_aic, min_ari
        ##re-run gmm for the min-bic case if compoenet > 1

        #self.gmm_min_comp, self.gmm_min_bic, self.gmm_min_aic, self.gmm_predicted_labels = \
        #         min_comp,          min_bic,          min_aic,      min_predicted_labels
        min_comp = kh
        min_bic, min_aic, min_ari, min_predicted_labels = self.runIIGMMClusteringForKnownComponents(kh, coverience_type, 50)

        return min_comp, min_bic, min_aic, min_ari, min_predicted_labels



def getLabel(i, labels):

	return labels[i]

def getallsimilarlabels(label, labels):
	simlabels = []
	for i in range(len(labels)):
		if labels[i] ==label:
			simlabels.append(i)
	return simlabels

	
def createTrainingDataAndClusteringObject1():
    
    tx_power_dBm = np.array([22.0, 25.0, 30.0, 28.0, 29.0])

    tx_loc = np.array([ [500, 500],
                        [400, 400],
                        [300, 200],
                        [800, 800],
                        [700, 600]
                      ])

    n_sample_per_config = 500
    dim_ratio = 0.1/100.0
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = tx_loc,
                              d_0_meter=10.0,
                              sigma_sq_db = 5.0
                              )

    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()

    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1

    n_sample =  n_sample_per_config * total_distinct_config
    #print "#-of generated samples: ",n_sample
    gsm.generateTrainingData(n_sample=n_sample, dim_ratio = dim_ratio, add_noise=True)
    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels)

    gsm.n_samples = n_sample
    gsm.n_sample_per_config = n_sample_per_config
    return  gsm, smc


def createTrainingDataAndClusteringObject2():
    tx_power_dBm = np.array([22.0, 25.0, 30.0, 28.0, 29.0])

    tx_loc = np.array([ [500, 500],
                        [400, 400],
                        [300, 200],
                        [800, 800],
                        [700, 600]
                      ])

    n_sample_per_config = 500
    dim_ratio = 0.2/100.0
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = tx_loc,
                              d_0_meter=10.0,
                              sigma_sq_db = 5.0
                              )

    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()

    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1

    n_sample =  n_sample_per_config * total_distinct_config
    #print "#-of generated samples: ",n_sample
    gsm.generateTrainingData(n_sample=n_sample, dim_ratio = dim_ratio, add_noise=True)
    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels)

    gsm.n_samples = n_sample
    gsm.n_sample_per_config = n_sample_per_config
    return  gsm, smc


def createTrainingDataAndClusteringObject3():
    tx_power_dBm = np.array([22.0, 25.0, 30.0, 28.0, 29.0])

    tx_loc = np.array([ [50, 50],
                        [40, 40],
                        [30, 20],
                        [80, 80],
                        [70, 60]
                      ])

    n_sample_per_config = 100
    dim_ratio = 0.5/100.0
    gsm = GenerateSpectrumMap(max_x_meter = 100.0,
                              max_y_meter = 100.0,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = tx_loc,
                              d_0_meter=10.0,
                              sigma_sq_db = 5.0
                              )

    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()

    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1

    n_sample =  n_sample_per_config * total_distinct_config
    #print "#-of generated samples: ",n_sample
    gsm.generateTrainingData(n_sample=n_sample, dim_ratio = dim_ratio, add_noise=True)
    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels)

    gsm.n_samples = n_sample
    gsm.n_sample_per_config = n_sample_per_config
    return  gsm, smc

def createTrainingDataAndClusteringObject4():
    tx_power_dBm = np.array([22.0, 25.0, 30.0, 28.0, 29.0])

    tx_loc = np.array([ [500, 500],
                        [400, 400],
                        [300, 200],
                        [800, 800],
                        [700, 600]
                      ])

    n_sample_per_config = 100
    dim_ratio = 0.4/100.0
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = tx_loc,
                              d_0_meter=10.0,
                              sigma_sq_db = 5.0
                              )

    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()

    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1

    n_sample =  n_sample_per_config * total_distinct_config
    #print "#-of generated samples: ",n_sample
    gsm.generateTrainingData(n_sample=n_sample, dim_ratio = dim_ratio, add_noise=True)
    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels)

    gsm.n_samples = n_sample
    gsm.n_sample_per_config = n_sample_per_config
    return  gsm, smc

def createTrainingDataAndClusteringObject5():
    tx_power_dBm = np.array([22.0, 25.0, 30.0, 28.0, 29.0])

    tx_loc = np.array([ [500, 500],
                        [400, 400],
                        [300, 200],
                        [800, 800],
                        [700, 600]
                      ])

    n_sample_per_config = 100
    dim_ratio = 0.5/100.0
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = tx_loc,
                              d_0_meter=10.0,
                              sigma_sq_db = 5.0
                              )

    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()

    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1

    n_sample =  n_sample_per_config * total_distinct_config
    #print "#-of generated samples: ",n_sample
    gsm.generateTrainingData(n_sample=n_sample, dim_ratio = dim_ratio, add_noise=True)
    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels)

    gsm.n_samples = n_sample
    gsm.n_sample_per_config = n_sample_per_config
    return  gsm, smc


def createTrainingDataAndClusteringObject6():
    tx_power_dBm = np.array([22.0, 25.0, 30.0, 28.0, 29.0])

    tx_loc = np.array([ [500, 500],
                        [400, 400],
                        [300, 200],
                        [800, 800],
                        [700, 600] 
                      ])   
    
    n_sample_per_config = 500
    dim_ratio = 0.6/100.0
    gsm = GenerateSpectrumMap(max_x_meter = 1000.0,
                              max_y_meter = 1000.0,
                              tx_power_dBm = tx_power_dBm,
                              tx_loc = tx_loc,
                              d_0_meter=10.0,
                              sigma_sq_db = 5.0
                              )
    gsm.generateIndividualMap()
    gsm.generateAllCombinationMap()

    total_distinct_config = int(np.power(2, tx_power_dBm.shape[0])) - 1

    n_sample =  n_sample_per_config * total_distinct_config
    #print "#-of generated samples: ",n_sample
    gsm.generateTrainingData(n_sample=n_sample, dim_ratio = dim_ratio, add_noise=True)
    smc = SpecMapClustering(x_indices = gsm.training_data_x_indx,
                            y_indices = gsm.training_data_y_indx,
                            vals = gsm.training_data_vals,
                            labels = gsm.training_data_labels)

    gsm.n_samples = n_sample
    gsm.n_sample_per_config = n_sample_per_config
    return  gsm, smc




def iteration1():
    file = open("tx_5_pt_30_n_10_td_1000_all_dim1.txt","w")
###1st iteration###--------------
    gsm, smc = createTrainingDataAndClusteringObject1()
    np.random.seed(1009993)
    #n_component_list = [4,7]
    #print 'gsm.al maps size: ', len(gsm.all_maps)

##-----------------------IEM Experiment------------------------------------------#
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    n_component_list = [31]
    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = smc.runGMMClustering(n_component_list)#(n_component_list = (np.array(smc.cur_training_matrix).shape[0]))
    iem_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    											      iem_derived_maps)
    print "ARI (IEM): ",  iem_min_ari
    print "Average Map Error (IEM): ", iem_avg_map_error



#-------------------EMII Experiment---------------------------------#
    #smc.preprocessTrainingDataEMII(pca_var_ratio = 0.95)
    #emii_min_comp, emii_min_bic, emii_min_aic, emii_min_ari, emii_predicted_labels = smc.runGMMClusteringII(n_component_list)#(n_component_list = np.array(smc.cur_training_matrix).shape[0])#n_component_list)#runIIGMMClusteringForKnownComponents()#runGMMClusteringII(n_component_list = n_component_list)
  
    #emii_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
    #                                           max_y = gsm.get_map_grid_y(),
    #                                           predicted_labels=emii_predicted_labels)
    #emii_real_map_indices, emii_derived_map_indices, emii_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    #                                                                                               emii_derived_maps)

#---#----------------K-Means Clustering Experiment---------------------------------#
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    min_comp = min(emii_min_comp, iem_min_comp)
    kms_predicted_labels, kms_ari = smc.runKMeansClusteringForKnownComponents(n_components = min_comp)
    kms_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=kms_predicted_labels)
    kms_real_map_indices, kms_derived_map_indices,kms_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              kms_derived_maps)
    print "ARI (K-Means): ",  kms_ari
    print "Average Map Error (K-Means): ", kms_avg_map_error

    file.write(str(0.1)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error))
    #file.write(str(7)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    file.close()

    #print "selected model component# ",min_comp
    #print "selected model components emii vs iem: ", emii_min_comp, iem_min_comp 
    #print "ARI (EMII vs IEM vs KMeans): ", emii_min_ari, " ", iem_min_ari, " ", kms_ari
    #print "Average Map Error (EMII vs IEM vs KMeans): ", emii_avg_map_error, " ", iem_avg_map_error, " ", kms_avg_map_error
    #file.write(str(5)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    #file.close()

#---------------------Display Maps--------------------------------------------------#
#--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])

    #for j in kms_derived_map_indices:
    #    kms_display_mapList.append( kms_derived_maps[j] )

    #gsm.displayMaps(map_list = kms_display_mapList, figFilename = './plots/kmeans_01_dim.png', n_rows=2)
#---##---------------display the heat maps for GMM IEM-----------------------
    #iem_display_mapList = []
    #for i in iem_real_map_indices:
    #    iem_display_mapList.append(gsm.all_maps[i])

    #for j in iem_derived_map_indices:
    #    iem_display_mapList.append( iem_derived_maps[j] )

    #gsm.displayMaps(map_list=iem_display_mapList, figFilename = './plots/iem_01_dim.png', n_rows=2)

#---##---------------display the heat maps for GMM EMII-----------------------
    #emii_display_mapList = []
    #for i in emii_real_map_indices:
    #    emii_display_mapList.append(gsm.all_maps[i])

    #for j in emii_derived_map_indices:
    #    emii_display_mapList.append( emii_derived_maps[j] )

    #gsm.displayMaps(map_list=emii_display_mapList, figFilename = './plots/emii_01_dim.png', n_rows=2)


def iteration2():
######2nd iteration####-------------
    file = open("tx_5_pt_30_n_10_td_1000_all_dim2.txt","w")
    gsm, smc = createTrainingDataAndClusteringObject2()
    np.random.seed(1009993)
    #n_component_list = [4,7]
    #print 'gsm.al maps size: ', len(gsm.all_maps)

##-----------------------IEM Experiment------------------------------------------#
    n_component_list = [31]
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = smc.runGMMClustering(n_component_list)#(n_component_list = (np.array(smc.cur_training_matrix).shape[0]))
    iem_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)
    print "ARI (IEM): ",  iem_min_ari
    print "Average Map Error (IEM): ", iem_avg_map_error



#-------------------EMII Experiment---------------------------------#
    #smc.preprocessTrainingDataEMII(pca_var_ratio = 0.95)
    #emii_min_comp, emii_min_bic, emii_min_aic, emii_min_ari, emii_predicted_labels = smc.runGMMClusteringII(n_component_list)#(n_component_list = np.array(smc.cur_training_matrix).shape[0])#n_component_list)#runIIGMMClusteringForKnownComponents()#runGMMClusteringII(n_component_list = n_component_list)

    #emii_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
    #                                           max_y = gsm.get_map_grid_y(),
    #                                           predicted_labels=emii_predicted_labels)
    #emii_real_map_indices, emii_derived_map_indices, emii_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    #                                                                                                   emii_derived_maps)

#---#----------------K-Means Clustering Experiment---------------------------------#
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    min_comp = min(emii_min_comp, iem_min_comp)
    kms_predicted_labels, kms_ari = smc.runKMeansClusteringForKnownComponents(n_components = min_comp)
    kms_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=kms_predicted_labels)
    kms_real_map_indices, kms_derived_map_indices,kms_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              kms_derived_maps)
    print "ARI (K-Means): ",  kms_ari
    print "Average Map Error (K-Means): ", kms_avg_map_error

    file.write(str(0.2)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error))
    #file.write(str(7)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    file.close()


    #print "selected model component# ",min_comp
    #print "selected model components emii vs iem: ", emii_min_comp, iem_min_comp
    #print "ARI (EMII vs IEM vs KMeans): ", emii_min_ari, " ", iem_min_ari, " ", kms_ari
    #print "Average Map Error (EMII vs IEM vs KMeans): ", emii_avg_map_error, " ", iem_avg_map_error, " ", kms_avg_map_error
    #file.write(str(6)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    #file.close()

#---------------------Display Maps--------------------------------------------------#
#--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])

    #for j in kms_derived_map_indices:
    #    kms_display_mapList.append( kms_derived_maps[j] )

    #gsm.displayMaps(map_list = kms_display_mapList, figFilename = './plots/kmeans_02_dim.png', n_rows=2)
#---##---------------display the heat maps for GMM IEM-----------------------
    #iem_display_mapList = []
    #for i in iem_real_map_indices:
    #    iem_display_mapList.append(gsm.all_maps[i])

    #for j in iem_derived_map_indices:
    #    iem_display_mapList.append( iem_derived_maps[j] )

    #gsm.displayMaps(map_list=iem_display_mapList, figFilename = './plots/iem_02_dim.png', n_rows=2)

#---##---------------display the heat maps for GMM EMII-----------------------
    #emii_display_mapList = []
    #for i in emii_real_map_indices:
    #    emii_display_mapList.append(gsm.all_maps[i])

    #for j in emii_derived_map_indices:
    #    emii_display_mapList.append( emii_derived_maps[j] )

    #gsm.displayMaps(map_list=emii_display_mapList, figFilename = './plots/emii_02_dim.png', n_rows=2)



def iteration3():
    file = open("tx_5_pt_30_n_10_td_1000_all_dim5_emii_new.txt","w")
####3rd iteration-------------###
    gsm, smc = createTrainingDataAndClusteringObject3()
    np.random.seed(1009993)
    #n_component_list = [4,7]
    #print 'gsm.al maps size: ', len(gsm.all_maps)

##-----------------------IEM Experiment------------------------------------------#
    n_component_list = [31]
    #smc.preprocessTrainingData(pca_var_ratio = 0.90)
    #iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = smc.runGMMClustering(n_component_list)#(n_component_list = (np.array(smc.cur_training_matrix).shape[0]))
    #iem_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
    #                                           max_y = gsm.get_map_grid_y(),
    #                                           predicted_labels=iem_predicted_labels)
    #iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    #                                                                                          iem_derived_maps)
    #print "ARI (IEM): ",  iem_min_ari
    #print "Average Map Error (IEM): ", iem_avg_map_error

#-------------------EMII Experiment---------------------------------#
    smc.preprocessTrainingDataEMII(pca_var_ratio = 0.90)
    emii_min_comp, emii_min_bic, emii_min_aic, emii_min_ari, emii_predicted_labels = smc.runGMMClusteringII(n_component_list)#(n_component_list = np.array(smc.cur_training_matrix).shape[0])#n_component_list)#runIIGMMClusteringForKnownComponents()#runGMMClusteringII(n_component_list = n_component_list)

    print 'emii_predicted_labels : ', emii_predicted_labels 
    emii_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=emii_predicted_labels)
    emii_real_map_indices, emii_derived_map_indices, emii_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                                   emii_derived_maps)
    print "ARI (EMII): ", emii_min_ari
    print "Average Map Error (EMII ): ", emii_avg_map_error

#---#----------------K-Means Clustering Experiment---------------------------------#
    #smc.preprocessTrainingData(pca_var_ratio = 0.90)
    #min_comp = 31#min(emii_min_comp, iem_min_comp)
    #kms_predicted_labels, kms_ari = smc.runKMeansClusteringForKnownComponents(n_components = min_comp)
    #kms_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
    #                                           max_y = gsm.get_map_grid_y(),
    #                                           predicted_labels=kms_predicted_labels)
    #kms_real_map_indices, kms_derived_map_indices,kms_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    #                                                                                          kms_derived_maps)
    #print "ARI (K-Means): ",  kms_ari
    #print "Average Map Error (K-Means): ", kms_avg_map_error

    ##print "selected model component# ",min_comp
    ##print "selected model components emii vs iem: ", emii_min_comp, iem_min_comp
    #print "ARI (EMII vs IEM vs KMeans): ", emii_min_ari, " ", iem_min_ari, " ", kms_ari
    #print "Average Map Error (EMII vs IEM vs KMeans): ", emii_avg_map_error, " ", iem_avg_map_error, " ", kms_avg_map_error
    file.write(str(0.5)+","+str(emii_min_ari)+"," +str(emii_avg_map_error))
    ##file.write(str(7)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    file.close()

#---------------------Display Maps--------------------------------------------------#
#--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])

    #for j in kms_derived_map_indices:
    #    kms_display_mapList.append( kms_derived_maps[j] )

    #gsm.displayMaps(map_list = kms_display_mapList, figFilename = './plots/kmeans_03_dim.png', n_rows=2)
#---##---------------display the heat maps for GMM IEM-----------------------
    #iem_display_mapList = []
    #for i in iem_real_map_indices:
    #    iem_display_mapList.append(gsm.all_maps[i])

    #for j in iem_derived_map_indices:
    #    iem_display_mapList.append( iem_derived_maps[j] )

    #gsm.displayMaps(map_list=iem_display_mapList, figFilename = './plots/iem_03_dim.png', n_rows=2)

#---##---------------display the heat maps for GMM EMII-----------------------
    #emii_display_mapList = []
    #for i in emii_real_map_indices:
    #    emii_display_mapList.append(gsm.all_maps[i])

    #for j in emii_derived_map_indices:
    #    emii_display_mapList.append( emii_derived_maps[j] )

    #gsm.displayMaps(map_list=emii_display_mapList, figFilename = './plots/emii_03_dim.png', n_rows=2)



def iteration4():
####4th iteration######-----------
    file = open("tx_5_pt_30_n_10_td_1000_all_dim4.txt","w")
    gsm, smc = createTrainingDataAndClusteringObject4()
    np.random.seed(1009993)
    #n_component_list = [4,7]
    #print 'gsm.al maps size: ', len(gsm.all_maps)

##-----------------------IEM Experiment------------------------------------------#
    n_component_list = [31]
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = smc.runGMMClustering(n_component_list)#(n_component_list = (np.array(smc.cur_training_matrix).shape[0]))
    iem_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)
    print "ARI (IEM): ",  iem_min_ari
    print "Average Map Error (IEM): ", iem_avg_map_error

#-------------------EMII Experiment---------------------------------#
    #smc.preprocessTrainingDataEMII(pca_var_ratio = 0.95)
    #emii_min_comp, emii_min_bic, emii_min_aic, emii_min_ari, emii_predicted_labels = smc.runGMMClusteringII(n_component_list)#(n_component_list = np.array(smc.cur_training_matrix).shape[0])#n_component_list)#runIIGMMClusteringForKnownComponents()#runGMMClusteringII(n_component_list = n_component_list)

    #emii_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
    #                                           max_y = gsm.get_map_grid_y(),
    #                                           predicted_labels=emii_predicted_labels)
    #emii_real_map_indices, emii_derived_map_indices, emii_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    #                                                                                                   emii_derived_maps)

#---#----------------K-Means Clustering Experiment---------------------------------#
    #smc.preprocessTrainingData(pca_var_ratio = 0.90)
    min_comp = 31#min(emii_min_comp, iem_min_comp)
    kms_predicted_labels, kms_ari = smc.runKMeansClusteringForKnownComponents(n_components = min_comp)
    kms_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=kms_predicted_labels)
    kms_real_map_indices, kms_derived_map_indices,kms_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              kms_derived_maps)
    print "ARI (K-Means): ",  kms_ari
    print "Average Map Error (K-Means): ", kms_avg_map_error

    #print "selected model component# ",min_comp
    #print "selected model components emii vs iem: ", emii_min_comp, iem_min_comp
    #print "ARI (EMII vs IEM vs KMeans): ", emii_min_ari, " ", iem_min_ari, " ", kms_ari
    #print "Average Map Error (EMII vs IEM vs KMeans): ", emii_avg_map_error, " ", iem_avg_map_error, " ", kms_avg_map_error
    file.write(str(0.4)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error))
    #file.write(str(8)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    file.close()
 
#---------------------Display Maps--------------------------------------------------#
#--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])

    #for j in kms_derived_map_indices:
    #    kms_display_mapList.append( kms_derived_maps[j] )

    #gsm.displayMaps(map_list = kms_display_mapList, figFilename = './plots/kmeans_04_dim.png', n_rows=2)
#---##---------------display the heat maps for GMM IEM-----------------------
    #iem_display_mapList = []
    #for i in iem_real_map_indices:
    #    iem_display_mapList.append(gsm.all_maps[i])

    #for j in iem_derived_map_indices:
    #    iem_display_mapList.append( iem_derived_maps[j] )

    #gsm.displayMaps(map_list=iem_display_mapList, figFilename = './plots/iem_04_dim.png', n_rows=2)

#---##---------------display the heat maps for GMM EMII-----------------------
    #emii_display_mapList = []
    #for i in emii_real_map_indices:
    #    emii_display_mapList.append(gsm.all_maps[i])

    #for j in emii_derived_map_indices:
    #    emii_display_mapList.append( emii_derived_maps[j] )

    #gsm.displayMaps(map_list=emii_display_mapList, figFilename = './plots/emii_04_dim.png', n_rows=2)

def iteration5():
    file = open("tx_5_pt_30_n_10_td_1000_all_dim5.txt","w")
####5th iteration######-----------
    gsm, smc = createTrainingDataAndClusteringObject5()
    np.random.seed(1009993)
    #n_component_list = [4,7]
    #print 'gsm.al maps size: ', len(gsm.all_maps)

##-----------------------IEM Experiment------------------------------------------#
    n_component_list = [31]
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = smc.runGMMClustering(n_component_list)#(n_component_list = (np.array(smc.cur_training_matrix).shape[0]))
    iem_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)
    print "ARI (IEM): ",  iem_min_ari										      
    print "Average Map Error (IEM): ", iem_avg_map_error

#-------------------EMII Experiment---------------------------------#
    #smc.preprocessTrainingDataEMII(pca_var_ratio = 0.95)
    #emii_min_comp, emii_min_bic, emii_min_aic, emii_min_ari, emii_predicted_labels = smc.runGMMClusteringII(n_component_list)#(n_component_list = np.array(smc.cur_training_matrix).shape[0])#n_component_list)#runIIGMMClusteringForKnownComponents()#runGMMClusteringII(n_component_list = n_component_list)

    #emii_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
    #                                           max_y = gsm.get_map_grid_y(),
    #                                           predicted_labels=emii_predicted_labels)
    #emii_real_map_indices, emii_derived_map_indices, emii_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    #                                                                                               emii_derived_maps)
    #print "ARI (EMII vs IEM): ", emii_min_ari, " ", iem_min_ari
    #print "Average Map Error (EMII vs IEM): ", emii_avg_map_error, " ", iem_avg_map_error

#---#----------------K-Means Clustering Experiment---------------------------------#
    #smc.preprocessTrainingData(pca_var_ratio = 0.90)
    min_comp = 31#min(emii_min_comp, iem_min_comp)
    kms_predicted_labels, kms_ari = smc.runKMeansClusteringForKnownComponents(n_components = min_comp)
    kms_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=kms_predicted_labels)
    kms_real_map_indices, kms_derived_map_indices,kms_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              kms_derived_maps)
    print "ARI (K-Means): ",  kms_ari
    print "Average Map Error (K-Means): ", kms_avg_map_error
    #print "selected model component# ",min_comp
    #print "selected model components emii vs iem: ", emii_min_comp, iem_min_comp
    #print "ARI (EMII vs IEM vs KMeans): ", emii_min_ari, " ", iem_min_ari, " ", kms_ari
    #print "Average Map Error (EMII vs IEM vs KMeans): ", emii_avg_map_error, " ", iem_avg_map_error, " ", kms_avg_map_error
    file.write(str(0.5)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error))
    #file.write(str(9)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    file.close()


#---------------------Display Maps--------------------------------------------------#



#--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])

    #for j in kms_derived_map_indices:
    #    kms_display_mapList.append( kms_derived_maps[j] )

    #gsm.displayMaps(map_list = kms_display_mapList, figFilename = './plots/kmeans_05_dim.png', n_rows=2)
#---##---------------display the heat maps for GMM IEM-----------------------
    #iem_display_mapList = []
    #for i in iem_real_map_indices:
    #    iem_display_mapList.append(gsm.all_maps[i])

    #for j in iem_derived_map_indices:
    #    iem_display_mapList.append( iem_derived_maps[j] )

    #gsm.displayMaps(map_list=iem_display_mapList, figFilename = './plots/iem_05_dim.png', n_rows=2)

#---##---------------display the heat maps for GMM EMII-----------------------
    #emii_display_mapList = []
    #for i in emii_real_map_indices:
    #    emii_display_mapList.append(gsm.all_maps[i])

    #for j in emii_derived_map_indices:
    #    emii_display_mapList.append( emii_derived_maps[j] )

    #gsm.displayMaps(map_list=emii_display_mapList, figFilename = './plots/emii_05_dim.png', n_rows=2)











def iteration6():
    file = open("tx_5_pt_30_n_10_td_1000_all_dim6.txt","w")
####6th iteration######-----------
    gsm, smc = createTrainingDataAndClusteringObject6()
    np.random.seed(1009993)
    #n_component_list = [4,7]
    #print 'gsm.al maps size: ', len(gsm.all_maps)

##-----------------------IEM Experiment------------------------------------------#
    n_component_list = [31]
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    iem_min_comp, iem_min_bic, iem_min_aic, iem_min_ari, iem_predicted_labels = smc.runGMMClustering(n_component_list)#(n_component_list = (np.array(smc.cur_training_matrix).shape[0]))
    iem_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=iem_predicted_labels)
    iem_real_map_indices, iem_derived_map_indices, iem_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              iem_derived_maps)
    print "ARI (IEM): ",  iem_min_ari
    print "Average Map Error (IEM): ", iem_avg_map_error

#-------------------EMII Experiment---------------------------------#
    #smc.preprocessTrainingDataEMII(pca_var_ratio = 0.95)
    #emii_min_comp, emii_min_bic, emii_min_aic, emii_min_ari, emii_predicted_labels = smc.runGMMClusteringII(n_component_list)#(n_component_list = np.array(smc.cur_training_matrix).shape[0])#n_component_list)#runIIGMMClusteringForKnownComponents()#runGMMClusteringII(n_component_list = n_component_list)

    #emii_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
    #                                           max_y = gsm.get_map_grid_y(),
    #                                           predicted_labels=emii_predicted_labels)
    #emii_real_map_indices, emii_derived_map_indices, emii_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
    #                                                                                               emii_derived_maps)

#---#----------------K-Means Clustering Experiment---------------------------------#
    smc.preprocessTrainingData(pca_var_ratio = 0.90)
    min_comp = 31#min(emii_min_comp, iem_min_comp)
    kms_predicted_labels, kms_ari = smc.runKMeansClusteringForKnownComponents(n_components = min_comp)
    kms_derived_maps = smc.generateDerivedMaps(max_x = gsm.get_map_grid_x(),
                                               max_y = gsm.get_map_grid_y(),
                                               predicted_labels=kms_predicted_labels)
    kms_real_map_indices, kms_derived_map_indices,kms_avg_map_error = smc.avgPairwiseMapError(gsm.all_maps,
                                                                                              kms_derived_maps)
    print "ARI (K-Means): ",  kms_ari
    print "Average Map Error (K-Means): ", kms_avg_map_error


    #print "selected model component# ",min_comp
    #print "selected model components emii vs iem: ", emii_min_comp, iem_min_comp
    #print "ARI (EMII vs IEM vs KMeans): ", emii_min_ari, " ", iem_min_ari, " ", kms_ari
    #print "Average Map Error (EMII vs IEM vs KMeans): ", emii_avg_map_error, " ", iem_avg_map_error, " ", kms_avg_map_error
    file.write(str(0.6)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error))
    #file.write(str(10)+","+str(kms_ari)+"," +str(iem_min_ari)+ ","+str(emii_min_ari)+ ","+str(kms_avg_map_error)+","+str(iem_avg_map_error)+ ","+str(emii_avg_map_error))
    file.close()

#---------------------Display Maps--------------------------------------------------#
    #--------display the heat maps for KMeans---------------------------
    #kms_display_mapList = []
    #for i in kms_real_map_indices:
    #    kms_display_mapList.append(gsm.all_maps[i])

    #for j in kms_derived_map_indices:
    #    kms_display_mapList.append( kms_derived_maps[j] )

    #gsm.displayMaps(map_list = kms_display_mapList, figFilename = './plots/kmeans_06_dim.png', n_rows=2)
#---##---------------display the heat maps for GMM IEM-----------------------
    #iem_display_mapList = []
    #for i in iem_real_map_indices:
    #    iem_display_mapList.append(gsm.all_maps[i])

    #for j in iem_derived_map_indices:
    #    iem_display_mapList.append( iem_derived_maps[j] )

    #gsm.displayMaps(map_list=iem_display_mapList, figFilename = './plots/iem_06_dim.png', n_rows=2)

#---##---------------display the heat maps for GMM EMII-----------------------
    #emii_display_mapList = []
    #for i in emii_real_map_indices:
    #    emii_display_mapList.append(gsm.all_maps[i])

    #for j in emii_derived_map_indices:
    #    emii_display_mapList.append( emii_derived_maps[j] )

    #gsm.displayMaps(map_list=emii_display_mapList, figFilename = './plots/emii_06_dim.png', n_rows=2)


if __name__ == '__main__':
    #file = open("pt_30_n_10_dim_0-3_td_1000_all_tx_correct.txt","w")
    #p1 = multiprocessing.Process(target=iteration1)
    #p2 = multiprocessing.Process(target=iteration2)
    p3 = multiprocessing.Process(target=iteration3)
    #p4 = multiprocessing.Process(target=iteration4)
    #p5 = multiprocessing.Process(target=iteration5)
    #p6 = multiprocessing.Process(target=iteration6)

    #p1.start()
    #p2.start()
    p3.start()
    #p4.start()
    #p5.start()
    #p6.start()

    #p1.join()
    #p2.join()
    p3.join()
    #p4.join()
    #p5.join()
    #p6.join()

    print("Done!")

