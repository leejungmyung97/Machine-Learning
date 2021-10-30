import pandas as pd
import numpy as np
import pprint
import matplotlib.pylab as plt
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall;
import time

# sum of distance for elbow method
kmeans_sumofDistance = {}

# silhouette
kmeans_silhouette = {}
gmm_silhouette = {}
clarans_silhouette = {}
meanshift_silhouette = {}
dbscan_silhouette = {}

# purity
kmeans_purity = {}
gmm_purity = {}
clarans_purity = {}
meanshift_purity = {}
dbscan_purity = {}


def main():
    # hyperparameter
    n_cluster = list(range(2, 13))
    DBSCAN_list = {'eps': [0.1, 0.2, 0.5, 5, 10, 100, 1000], 'min_sample': [10, 20]}
    MeanShift_list = [None, 1.0, 2.0, 10, 100]
    MeanShift_list_plot = [0, 1.0, 2.0, 10, 100]

    print("=== 1. Data Load & Missing Data check")
    dataset = pd.read_csv('Lab2_PHW2\housing.csv')  # load dataset
    # print(dataset.describe())
    # print(dataset.isna().sum())
    # print(dataset.info())

    print("=== 2. split median_house_value & labeling")
    median_house_value = pd.DataFrame(dataset["median_house_value"])
    dataset = dataset.drop(columns=["median_house_value"])

    bins = list(range(14998, 500002, 48500))
    median_house_value['label'] = pd.cut(median_house_value["median_house_value"],
                                         bins,
                                         labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # print(median_house_value.groupby('label')['median_house_value'].apply(my_summary).unstack())

    print("=== 3. drop not use data (total_bedrooms)")
    dataset = dataset.drop(columns=["total_bedrooms"])
    # print(dataset.info())

    print("=== 4. Preprocessing")
    pre_feature = Preprocessing(dataset, ["ocean_proximity"],
                                ["longitude", "latitude", "housing_median_age", "total_rooms", "population",
                                 "households", "median_income"])
    # pprint.pprint(pre_feature)

    print("=== 5. make clustering")
    for key, value in pre_feature.items():
        FindBestCombination(key, value, n_cluster, DBSCAN_list, MeanShift_list, median_house_value['label'])

    print("=== 6. Result")
    # check sum of distance for elbow method
    makeplot("KMeans_distance", kmeans_sumofDistance, n_cluster)

    # #silhouette
    makeplot("KMeans_silhouette", kmeans_silhouette, n_cluster)
    makeplot("EM_silhouette", gmm_silhouette, n_cluster)
    makeplot("DBSCAN_silhouette", dbscan_silhouette, DBSCAN_list['eps'])
    makeplot("MeanShift_distance", meanshift_silhouette, MeanShift_list_plot)
    makeplot("Clarans_distance", clarans_silhouette, n_cluster)

    key, value = fineMaxValueKey(kmeans_silhouette)
    print("k-means best silhouette : ", value, key)
    key, value = fineMaxValueKey(gmm_silhouette)
    print("EM best silhouette : ", value, key)
    key, value = fineMaxValueKey(dbscan_silhouette)
    print("DBSCAN best silhouette : ", value, key)
    key, value = fineMaxValueKey(meanshift_silhouette)
    print("MeanShift best silhouette : ", value, key)
    key, value = fineMaxValueKey(clarans_silhouette)
    print("Clarans best silhouette : ", value, key)

    # purity
    makeplot("KMeans_purity", kmeans_purity, n_cluster)
    makeplot("EM_purity", gmm_purity, n_cluster)
    makeplot("DBSCAN_purity", dbscan_purity, DBSCAN_list['eps'])
    makeplot("MeanShift_distance", meanshift_purity, MeanShift_list_plot)
    makeplot("Clarans_distance", clarans_purity, n_cluster)

    key, value = fineMaxValueKey(kmeans_purity)
    print("k-means best purity : ", value, key)
    key, value = fineMaxValueKey(gmm_purity)
    print("k-means best purity : ", value, key)
    key, value = fineMaxValueKey(dbscan_purity)
    print("DBSCAN best purity : ", value, key)
    key, value = fineMaxValueKey(meanshift_purity)
    print("MeanShift best purity : ", value, key)
    key, value = fineMaxValueKey(clarans_purity)
    print("Clarans best purity : ", value, key)


# for one-hot-encoding
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


def Preprocessing(feature, encode_list, scale_list):
    # feature : dataframe of feature

    # scaler
    scaler_stndard = preprocessing.StandardScaler()
    scaler_MM = preprocessing.MinMaxScaler()
    scaler_robust = preprocessing.RobustScaler()
    scaler_maxabs = preprocessing.MaxAbsScaler()
    scaler_normalize = preprocessing.Normalizer()
    scalers = [None, scaler_stndard, scaler_MM, scaler_robust, scaler_maxabs, scaler_normalize]
    scalers_name = ["original", "standard", "minmax", "robust", "maxabs", "normalize"]

    # encoder
    encoder_ordinal = preprocessing.OrdinalEncoder()
    # one hot encoding => using pd.get_dummies() (not used preprocessing.OneHotEncoder())
    encoders_name = ["ordinal", "onehot"]

    # result box
    result_dictionary = {}
    i = 0

    if encode_list == []:
        for scaler in scalers:
            if i == 0:  # not scaling
                result_dictionary[scalers_name[i]] = feature.copy()

            else:
                # ===== scalers
                result_dictionary[scalers_name[i]] = feature.copy()
                result_dictionary[scalers_name[i]][scale_list] = scaler.fit_transform(feature[scale_list])  # scaling
            i = i + 1
        return result_dictionary

    for scaler in scalers:
        if i == 0:  # not scaling
            result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
            result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(
                feature[encode_list])
            result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
            result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"], encode_list)


# ===== scalers + ordinal encoding
    else:
        result_dictionary[scalers_name[i] + "_ordinal"] = feature.copy()
        result_dictionary[scalers_name[i] + "_ordinal"][scale_list] = scaler.fit_transform(feature[scale_list])
        result_dictionary[scalers_name[i] + "_ordinal"][encode_list] = encoder_ordinal.fit_transform(feature[encode_list])

# ===== scalers + OneHot encoding
        result_dictionary[scalers_name[i] + "_onehot"] = feature.copy()
        result_dictionary[scalers_name[i] + "_onehot"][scale_list] = scaler.fit_transform(feature[scale_list])
        result_dictionary[scalers_name[i] + "_onehot"] = dummy_data(result_dictionary[scalers_name[i] + "_onehot"], encode_list)
        i = i + 1
        return result_dictionary


def FindBestCombination(preprocessing_name, feature, n_cluster, DBSCAN_list, MeanShift_list, purity_GT):
    print(preprocessing_name)
    # n_cluster : list number of cluster (use in Kmeans, GMM, clarans)
    # clarans_list : list of clarans parameters (numlocal, maxneighbor)
    # DBSCAN_list : list of DBSCAN parameters (eps, min_sample)
    # MeanShift_list : list of MeanShift parameters (bandwidth)

    # KMeans
    print("Kmeans")
    start_time = time.time()
    kmean_sum_of_squared_distances = []
    kmean_silhouette_sub = []
    kmeans_purity_sub = []

    for k in n_cluster:
        kmeans = KMeans(n_clusters=k).fit(feature)

        # sum of distance for elbow methods
        kmean_sum_of_squared_distances.append(kmeans.inertia_)

        # silhouette (range -1~1)
        kmean_silhouette_sub.append(silhouette_score(feature, kmeans.labels_, metric='euclidean'))

        # purity
        kmeans_purity_sub.append(purity_score(purity_GT, kmeans.labels_))

    kmeans_sumofDistance[preprocessing_name] = kmean_sum_of_squared_distances
    kmeans_silhouette[preprocessing_name] = kmean_silhouette_sub
    kmeans_purity[preprocessing_name] = kmeans_purity_sub
    print(time.time() - start_time)

    # GaussianMixture (EM, GMM)
    print("EM")
    start_time = time.time()
    gmm_silhouette_sub = []
    gmm_purity_sub = []

    for k in n_cluster:
        gmm = GaussianMixture(n_components=k)
        labels = gmm.fit_predict(feature)
        gmm_silhouette_sub.append(silhouette_score(feature, labels, metric='euclidean'))
        gmm_purity_sub.append(purity_score(purity_GT, labels))

    gmm_silhouette[preprocessing_name] = gmm_silhouette_sub
    gmm_purity[preprocessing_name] = gmm_purity_sub
    print(time.time() - start_time)

    # clarans
    print("clarans")
    data = np.array(feature)
    data = data.tolist()
    clarans_silhouette_sub = []
    clarans_purity_sub = []

    max_sum = 0
    for k in n_cluster:
        clarans_instance = clarans(data, k, 6, 4)
        (ticks, result) = timedcall(clarans_instance.process)
        clusters = clarans_instance.get_clusters()
        label = clusterToIdx(clusters)
        clarans_silhouette_sub.append(silhouette_score(feature, label, metric='euclidean'))
        clarans_purity_sub.append(purity_score(purity_GT, label))

    clarans_silhouette[preprocessing_name] = clarans_silhouette_sub
    clarans_purity[preprocessing_name] = clarans_purity_sub

    # DBSCAN
    print("dbscan")
    start_time = time.time()
    dbscan_silhouette_sub = []
    dbscan_purity_sub = []

    for eps in DBSCAN_list["eps"]:
        max_silhouette = -2
        max_purity = -2

        for sample in DBSCAN_list["min_sample"]:
            dbscan = DBSCAN(eps=eps, min_samples=sample)
            label = dbscan.fit_predict(feature)

            try:
                current_silhouette = silhouette_score(feature, label, metric='euclidean')
            except:
                current_silhouette = -5

            if max_silhouette < current_silhouette:
                max_silhouette = current_silhouette

            current_purity = purity_score(purity_GT, label)
            if max_purity < current_purity:
                max_purity = current_purity

        dbscan_silhouette_sub.append(max_silhouette)
        dbscan_purity_sub.append(max_purity)

    dbscan_silhouette[preprocessing_name] = dbscan_silhouette_sub
    dbscan_purity[preprocessing_name] = dbscan_purity_sub
    print(time.time() - start_time)

    # meanShift
    print("meanshift")
    start_time = time.time()
    meanshift_silhouette_sub = []
    meanshift_purity_sub = []

    for bw in MeanShift_list:
        meanShift = MeanShift(bandwidth=bw)
        label = meanShift.fit_predict(feature)
        print(label)
        print(time.time() - start_time)

        try:
            current_silhouette = silhouette_score(feature, label, metric='euclidean')
        except:
            current_silhouette = -1
        meanshift_silhouette_sub.append(current_silhouette)
        meanshift_purity_sub.append(purity_score(purity_GT, label))

    meanshift_silhouette[preprocessing_name] = meanshift_silhouette_sub
    meanshift_purity[preprocessing_name] = meanshift_purity_sub
    print(time.time() - start_time)


# Test purity
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def makeplot(title, dict, x_list):
    for key, value in dict.items():
        plt.plot(x_list, value, label=key)

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.tight_layout()
    plt.show()


def my_summary(x):
    result = {
        'sum': x.sum(),
        'count': x.count(),
        'mean': x.mean(),
        'variance': x.var()
    }
    return result


def fineMaxValueKey(dict):
    key = None
    largest = 0
    for keys, item in dict.items():
        if max(item) > largest:
            largest = max(item)
            key = keys

    return key, largest


def clusterToIdx(clusters):
    idx_list = [-1 for i in range(0, 20640)]
    idx = 0

    for k in clusters:
        for i in k:
            idx_list[i] = idx
        idx = idx + 1

    return idx_list


if __name__ == "__main__":
    main()
