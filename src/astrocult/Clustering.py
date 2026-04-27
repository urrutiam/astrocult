# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram

import math

from circular_clustering.circular_k_means_weighted import CircularWeightedKMeans

import seaborn as sns

class Clustering:
    '''
    Class containing methods for grouping sites together based on their associated data.
    '''
    
    def __create_bins(self,
                      df: pd.DataFrame,
                      declinations_column: str, 
                      bin_ranges: list=[-90, -23.5, -15, 0, 15, 23.5, 90]) -> pd.DataFrame:
        '''
        Asignes each declination value to a bin range. 
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data with declinations in the `declinations_column` column
        declinations_column : str
            The name of the column containing the declination values
        bin_ranges : list
            The bin ranges to which each declination will be assigned to.
            Default: [-90, -23.5, -15, 0, 15, 23.5, 90]. The first bin is [-90,-23.5) etc.
                                                                           
        Returns
        -------
            The original dataframe with the 'Bin' column attached to it
        Raises
        ------
        ValueError
            If `df` or the `declinations_column` is None
 
        '''
        
        if df is None or declinations_column is None:
            raise ValueError("`df` and `declinations_column` cannot be None.")
            
        df_bin = pd.DataFrame()
        df_bin['Bin'] = pd.cut(df[declinations_column], bin_ranges)
        
        return df_bin['Bin']

    def create_PCA(self,
                  df: pd.DataFrame,
                  no_components: int=2) -> tuple[pd.DataFrame, np.ndarray]:
        '''
        Performs a principal component analysis on the provided data by 
        redefining the dataframe in `no_components` dimensions. Each new dimension 
        will be a linear euqation of two or more initial dimensions (features)
        Data is normalized using `self.__normalize_data`.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing numerical data in all columns
        no_components : int
            The number of components (new dimensions) of the dataframe. Default: 2
            
        Returns
        -------
        Tuple
            The dataframe redefined in `no_components` features and the components as a ndarray
            
        Raises
        ------
        ValueError
            If the dataframe is None
        '''
        
        if df is None:
            raise ValueError("The dataframe cannot be None.")

        df_norm = self.__normalize_data(df)

        pca = PCA(n_components=no_components)
        principalComponents = pca.fit_transform(df_norm)
        principalDf = pd.DataFrame(data = principalComponents)
        
        print ("The variance of each selected component: ", pca.explained_variance_ratio_)
        
        return principalDf, pca.components_
    
    def show_PCA_correlation_matrix(self,
                                    components: list):
        '''
        Plots the PCA components' correlation matrix based on which the 
        correlation of each original feature in the new component can be assessed.
        The PCA are shown on the vertical axis while the original features on the 
        horizontal axis.
        
        Parameters
        ----------
        components : list
            The components returned by `self.create_PCA`
        Returns
        -------
        
        Raises
        ------
        ValueError
            If the argument is None
        '''
        
        if components is None:
            raise ValueError("Argument cannot be None.")
        
        plt.figure()
        sns.set(font_scale=0.7)
        sns.heatmap(components, annot=True)
        plt.show()
        
    def __circular_distance(self, angle1, angle2):
        '''
        Calculate the circular distance in radians between two angles. It is used for the circular KMeans algorithm instead of the classic Euclidian distance.
        
        Parameters
        ----------
        angle1 : float
            The angle in radians
        angle2 : float
            The angle in radians
            
        Returns
        -------
        pd.DataFrame
            The circular distance in radians between the angles using the arctan2 function
            
        Raises
        ------

        '''

        diff = angle1 - angle2
        return np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    
    def __normalize_data(self, 
                       df: pd.DataFrame) -> pd.DataFrame:
        '''
        Normalizes the dataframe data by using a sklearn.preprocessing.MinMaxScaler.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing numerical data in all columns
            
        Returns
        -------
        pd.DataFrame
            The normalized dataframe
            
        Raises
        ------
        ValueError
            If the dataframe is None or contains nonnumerical data
        '''
        
        if df is None:
            raise ValueError("The dataframe cannot be None.")
        
        if not df.select_dtypes(exclude='number').empty:
            raise ValueError("Only numerical data is accepted.")
        
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(df)
        
        df_norm = pd.DataFrame(x_scaled)
        
        return df_norm.set_axis(df.columns, axis=1)
    
    def get_optimal_number_of_kmeans_clusters(self,
                                    df: pd.DataFrame, 
                                    max_clusters: int=10) -> int:
        '''
        Calculates the optimal number of clusters by using the Silouette score.
        
        Parameters
        ----------
        df : pd.Dataframe
            The dataframe containing features as columms
        max_clusters : int
            The maximum number of clusters for which the score is 
            computed. Default: 10 
            
        Returns
        -------
        int
            The optimal number of clusters
            
        Raises
        ------
        ValueError
            If the dataframe is None
        '''
        
        if df is None:
            raise ValueError("The dataframe cannot be None.")

        sil_scores = list()
        
        df_norm = self.__normalize_data(df)
                
        for n_clusters in range(2, max_clusters):
          kmeans = KMeans(n_clusters=n_clusters, random_state=10)
          kmeans.fit(df_norm)
          
          sil_scores.append(silhouette_score(df, kmeans.labels_))
        
        return sil_scores.index(max(sil_scores))
    
    def create_kmeans_clusters(self, 
                                df: pd.DataFrame, 
                                n_clusters: int=10) -> tuple[KMeans, pd.DataFrame]:
        '''
        Groups the dataframe points into clusters based on the KMeans algorithm.
        Should be used for non circular data. For circular data use 'create_circular_kmeans_clusters' instead.
        Data is normalized using `self.__normalize_data`.
        
        Parameters
        ----------
        df : pd.Dataframe
            The dataframe containing features as columms
        max_clusters : int
            The maximum number of clusters generated. Default: 10 
            
        Returns
        -------
        Tuple
            sklearn.cluster.KMeans
                Model
            pd.Dataframe
                The dataframe with the appended column 'kmeans-clusters' representing the clusters
            
        Raises
        ------
        ValueError
            If the dataframe is None
        '''
        
        if df is None:
            raise ValueError("The dataframe cannot be None.")
        
        df_norm = self.__normalize_data(df)    
        
        model = KMeans(n_clusters=n_clusters, random_state=10)
        model.fit(df_norm)
        
        df2 = df.copy()
        
        df2['kmeans-clusters'] = model.labels_
          
        return model, df2

    def create_circular_kmeans_clusters(self, 
                                df: pd.DataFrame, 
                                circulardata_column,
                                n_clusters: int=10) -> tuple[CircularWeightedKMeans, pd.DataFrame]:
        '''
        Groups the dataframe points into clusters based on the circular KMeans algorithm.
        Use it instead of 'create_kmeans_clusters' when circular data such as azimuths needs to be clustered as the traditional KMeans does not know that 0 is 360 and can group in distinct clusters objects around these value.
        Data is not normalized.
        
        Parameters
        ----------
        df : pd.Dataframe
            The dataframe containing features as columms
        ciculardata_column : string
            The column containing circular data in degrees, e.g., azimuths
        max_clusters : int
            The maximum number of clusters generated. Default: 10 

            
        Returns
        -------
        Tuple
            circular_clustering.circular_k_means_weighted.CircularWeightedKMeans
                Model
            pd.Dataframe
                The dataframe with the appended column 'ckmeans-clusters' representing the clusters
            
        Raises
        ------
        ValueError
            If the dataframe is None
        '''    
        
        if df is None:
            raise ValueError("The dataframe cannot be None.")
        
        df2 = df.copy()
        
        # convert degrees to to [-pi, pi] interval
        df2[circulardata_column] = df2[circulardata_column].apply(lambda x: math.remainder(math.radians(x), 2 * math.pi))
                
        circXmeans = CircularWeightedKMeans(n_clusters)
        circXmeans.fit(df2[circulardata_column])

        # Get centroids
        centroids = circXmeans.centroids

      
        # Initialize a list to store cluster labels
        cluster_labels = []
        
        # Iterate over each data point to assign it to the closest centroid
        for az_value in df2[circulardata_column]:
            distances = [self.__circular_distance(az_value, c) for c in centroids]
            closest_centroid_index = np.argmin(distances)
            cluster_labels.append(closest_centroid_index)
        
        df3 = df.copy()
        df3['ckmeans-clusters'] = cluster_labels
        
        return circXmeans, df3
        
    def create_hierachical_clusters(self, 
                                    df: pd.DataFrame, 
                                    n_clusters: int=None, 
                                    distance_threshold: float=None, 
                                    plot_clusters: bool=True,
                                    metric: str="manhattan",
                                    linkage: str="complete",
                                    compute_distances: bool=True) -> AgglomerativeClustering:
        '''
        Groups the dataframe points into clusters using an Agglomerative clustering method.
        For now it uses `linkage=complete`. Data is normalized using `self.__normalize_data`.
        Parameters
        ----------
        df : pd.Dataframe
            The dataframe containing features as columms
        n_clusters : int
            The number of clusters to find. It must be None if distance_threshold is not None.
        distance_threshold : float
            The linkage distance threshold at or above which clusters will not 
            be merged. If not None, n_clusters must be None
            Setting `distance_threshold=0` ensures we compute the full tree
        metric : str
            The metric used to compute the distance. Default: manhattan. For 
            possible values see `sklearn.cluster.AgglomerativeClustering`
        linkage : str
            Cluster linkage mode. Default: complete. For possible values see 
            `sklearn.cluster.AgglomerativeClustering`
        compute_distances : bool
            Default: True. See `sklearn.cluster.AgglomerativeClustering` for
            details
        Returns
        -------
        Tuple
            sklearn.cluster.AgglomerativeClustering
                Model
            pd.Dataframe
                The dataframe with the appended column representing the clusters
                
        Raises
        ------
        ValueError
            If the dataframe is None or 
            distance_threshold is not None and n_clusters is not None
        '''
        
        if df is None:
            raise ValueError("The dataframe cannot be None.")

        df_norm = self.__normalize_data(df)
        
        model = AgglomerativeClustering(distance_threshold=distance_threshold, n_clusters=n_clusters, metric=metric, compute_distances=compute_distances, linkage=linkage)
        model = model.fit(df_norm)
         
        df2 = df.copy()
        
        df2['hierarchical-clusters'] = model.labels_
        
        return model, df2

    def show_clusters(self,
                      df: pd.DataFrame, 
                      axis_cols: list,
                      model: AgglomerativeClustering|KMeans):        
        '''
        Displays a 2D plot of the clusters using the `axis_cols` as columns.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the columns to be plotted
        axis_cols : list
            The 2 element list containig the columns representing the axes
        model : `sklearn.cluster.AgglomerativeClustering` or `sklearn.cluster.KMeans`

        Returns
        -------
        
        Raises
        ------
        ValueError
        
        '''
         
        if df is None or axis_cols is None or model is None:
            raise ValueError("Parameters cannot be None.")
        if type(model) == AgglomerativeClustering and type(model) == KMeans:
            raise ValueError("model can only be AgglomerativeClustering or Kmeans.")
        
        if (len(axis_cols) != 2):
          raise Exception("`axis_cols` must be 2.")
        
        cluster_col = None
        if type(model) == AgglomerativeClustering: 
            cluster_col = "hierarchical-clusters"  
        if type(model) == KMeans: 
            cluster_col = "kmeans-clusters"  
        if type(model) == CircularWeightedKMeans: 
            cluster_col = "ckmeans-clusters"  
            
        plt.scatter(df[axis_cols[0]], df[axis_cols[1]], c=df[cluster_col], cmap='rainbow')
        plt.legend(loc="best")
        plt.xlabel(axis_cols[0])
        plt.ylabel(axis_cols[1])
        plt.show()
            
        
    def show_dendrogram(self, 
                        model: AgglomerativeClustering, 
                        **kwargs):
        '''
        Displays the dendogram of the agglomerative clustering model.
        This function requires that the model is build with `compute_distances=True`.
        
        Parameters
        ----------
        model : sklearn.cluster.AgglomerativeClustering
        
        Returns
        -------
        
        Raises
        ------
        ValueError
        
        '''
        
        if type(model) != AgglomerativeClustering: 
            raise ValueError("`model` must be of type sklearn.cluster.AgglomerativeClustering")
        
        # Create linkage matrix and then plot the dendrogram
    
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count
    
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)
    
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
          
