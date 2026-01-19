# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
from typing import Tuple

class Clustering:
    '''
    '''
    
    def __create_bins(self,
                      df: pd.DataFrame,
                      declinations_column: str, 
                      bin_ranges: list=[-90, -23.5, -15, 0, 15, 23.5, 90]) -> pd.DataFrame:
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        
        Raises
        ------
        
        '''
               
        df_bin = pd.DataFrame()
        df_bin['Bin'] = pd.cut(df[declinations_column], bin_ranges)
        
        return df_bin['Bin']

    def createPCA(self,
                  df: pd.DataFrame,
                  no_components: int=2) -> Tuple:
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        
        Raises
        ------
        
        '''
        
        if df is None:
            return None

        pca = PCA(n_components=no_components)
        principalComponents = pca.fit_transform(df)
        principalDf = pd.DataFrame(data = principalComponents)
        
        print (pca.explained_variance_ratio_)
        
        return principalDf, pca.components_
    
    def normalize_data(self, 
                       df: pd.DataFrame) -> pd.DataFrame:
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        
        Raises
        ------
        
        '''
        
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(df)
        
        return pd.DataFrame(x_scaled)
    
    def get_optimal_number_of_kmeans_clusters(self,
                                    df: pd.DataFrame, 
                                    max_clusters: int, 
                                    show_plots:bool=True) -> int:
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        
        Raises
        ------
        
        '''
        
        inertia = list()
        sil_scores = list()
        for n_clusters in range(2, max_clusters):
          kmeans = KMeans(n_clusters=n_clusters, random_state=10)
          kmeans.fit(df)
          inertia.append(kmeans.inertia_)
          sil_scores.append(silhouette_score(df, kmeans.labels_))
          
        if show_plots:
          plt.plot(range(2, max_clusters), inertia)
          plt.title('Elbow Method')
          plt.xlabel('Number of clusters')
          plt.ylabel('Inertia')
          plt.show()
          
          plt.plot(range(2, max_clusters), sil_scores, color = 'salmon')
          plt.xlabel('Number of clusters')
          plt.ylabel('Silhouette score')
          plt.title('Silouette metrics')
          plt.axvline(x = sil_scores.index(max(sil_scores))+2, linestyle = 'dotted', color = 'red')
          plt.show()
          
        return sil_scores.index(max(sil_scores)) + 2
    
    def create__kmeans_clusters(self, 
                                df: pd.DataFrame, 
                                n_clusters: int) -> KMeans:
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        
        Raises
        ------
        
        '''
        
        model = KMeans(n_clusters=n_clusters, random_state=10)
        model.fit(df)
          
        return model
    
    def show_kmeans_clusters(self, 
                             df: pd.DataFrame, 
                             cluster_col: str, 
                             axis_cols: list, 
                             model: KMeans):
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        
        Raises
        ------
        
        '''
          
        if (len(axis_cols) != 2):
          raise Exception("Axis columns must be 2.")
          
        plt.scatter(df[axis_cols[0]], df[axis_cols[1]], c=df[cluster_col], cmap='rainbow')
        plt.legend(loc="best")
        plt.xlabel(axis_cols[0])
        plt.ylabel(axis_cols[1])
        plt.show()
    
    def create_hierachical_clusters(self, 
                                    df: pd.DataFrame, 
                                    n_clusters:int=3, 
                                    distance_threshold=None, 
                                    plot_clusters=True) -> AgglomerativeClustering:
        '''
        
        Parameters
        ----------
        # setting distance_threshold=0 ensures we compute the full tree.
        Returns
        -------
        
        Raises
        ------
        
        '''
        
        model = AgglomerativeClustering(distance_threshold=None, n_clusters=n_clusters, metric='manhattan', compute_distances=True, linkage='complete')
        model = model.fit(df)
          
        if plot_clusters == True:
          plt.figure(figsize=(70,10))
          self.plot_dendrogram(model, truncate_mode="level")
          ax = plt.gca()
          ax.tick_params(axis='x', which='major', labelsize=15)
          ax.tick_params(axis='y', which='major', labelsize=8)
          plt.xlabel("Number of declinations in cluster node (or index of declination if no parenthesis).")
          plt.show()
          
        return model
    
    def plot_dendrogram(self, 
                        model: AgglomerativeClustering, 
                        **kwargs):
        '''
        
        Parameters
        ----------
        
        Returns
        -------
        
        Raises
        ------
        
        '''
        
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