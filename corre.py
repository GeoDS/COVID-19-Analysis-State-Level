from tracemalloc import start
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
from kshape.core import kshape, zscore
from minepy import MINE
from scipy.stats import spearmanr
import geopandas as gpd
import argparse
import seaborn as sns
import scipy.stats
from scipy.signal import argrelextrema
import epyestim
import epyestim.covid19 as covid19
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
import bayesian_changepoint_detection.offline_likelihoods as offline_ll
from bayesian_changepoint_detection.priors import const_prior
from functools import partial
from sklearn.metrics import silhouette_score
from matplotlib.lines import Line2D

path="../2yrdata/"
cci=pd.read_csv(path+'two_years_cci.csv',index_col = "date").drop(["Alaska","Hawaii"],axis=1)
cmi=pd.read_csv(path+'two_years_cmi.csv',index_col = "date")
daily = pd.read_csv(path+"two_year_daily.csv",parse_dates=["date"],index_col="date").drop(["Alaska","Hawaii"],axis=1)
output_path = "../results/2yr/"
# cases=pd.read_csv(path+"2021_state_cases.csv",index_col="date")
# deaths=pd.read_csv(path+"2021_state_deaths.csv",index_col="date")
time_lags = [3,6,9,12,15,18,21]
plt.rcParams.update({'font.size': 22})


def get_mic(a,b):
    mine=MINE(est="mic_approx")
    mine.compute_score(a,b)
    return mine.mic()


def get_stats(a,b):
    mic = []
    # spearman = []
    for state in a.columns:
        x = a[state][6:]
        y = moving_average(b[state],7)
        m = get_mic(x,y)
        # r, p = spearmanr(x, y)
        mic.append(m)
        # spearman.append(r)
    return mic#,spearman

def mic_boxplot(bps,x=cci,y=cmi,z=daily,glob = False,fname = "temporal_boxplot.pdf"):
    cci_mic = pd.DataFrame(columns = bps[:-1])
    cmi_mic = pd.DataFrame(columns = bps[:-1])
    l = len(bps)
    for i in range(l-1):
        star = bps[i]
        end = bps[i+1]
        a = x[star:end]
        b = z[star:end]
        m1= get_stats(a,b)
        cci_mic[star] = m1

        c = y[star:end]
        m2 = get_stats(c,b)
        cmi_mic[star] = m2
    if glob:
        star = "2020-04-01"
        end = "2021-12-31"
        a = x[star:end]
        b = z[star:end]
        m1= get_stats(a,b)
        cci_mic["all"] = m1

        c = y[star:end]
        m2 = get_stats(c,b)
        cmi_mic["all"] = m2
    
    cci_mic = cci_mic.assign(idx_type="CCI")
    cmi_mic = cmi_mic.assign(idx_type="CMI")
    df = pd.concat([cci_mic,cmi_mic])
    mdf = pd.melt(df, id_vars=['idx_type'], var_name=['breakpoints'])
    fig,ax = plt.subplots(figsize=(12, 7))
    sns.boxplot(x="breakpoints", y="value", hue="idx_type", data=mdf,ax=ax) 
    plt.legend(title ="index type",loc=1)
    plt.ylabel("MIC Correlation")
    plt.xlabel("Start of Breakpoints")
    plt.yticks(np.arange(0.1, 1, step=0.1))
    plt.ylim(0.1,1.01)
    plt.savefig(output_path+fname,dpi=300,bbox_inches='tight')
    return df

def spatial_mic_boxplot(bps,labels,x=cci,y=cmi,z=daily,glob=False,fname = "spatial_boxplot.pdf"):
    k = len(labels["label"].unique())
    cci_mic = pd.DataFrame(columns = bps[:-1])
    cmi_mic = pd.DataFrame(columns = bps[:-1])
    l = len(bps)
    for i in range(l-1):
        star = bps[i]
        end = bps[i+1]
        a = x[star:end]
        b = z[star:end]
        m1= get_stats(a,b)
        cci_mic[star] = m1

        c = y[star:end]
        m2 = get_stats(c,b)
        cmi_mic[star] = m2
    if glob:
        star = "2020-04-01"
        end = "2021-12-31"
        a = x[star:end]
        b = z[star:end]
        m1= get_stats(a,b)
        cci_mic["all"] = m1

        c = y[star:end]
        m2 = get_stats(c,b)
        cmi_mic["all"] = m2


    #legends = {0:"north",1:"south"}
    cci_mic.index = x.columns
    cci_mic.index.name = "state"
    a2 = pd.merge(cci_mic,labels,left_on="state",right_on="state")
    a2.set_index("state",inplace=True)
    #a2["label"] = a2["label"].apply(lambda x:legends[x])
    cci_mdf = pd.melt(a2, id_vars=['label'], var_name=['breakpoints'])

    cmi_mic.index = x.columns
    cmi_mic.index.name = "state"
    b2 = pd.merge(cmi_mic,labels,left_on="state",right_on="state")
    b2.set_index("state",inplace=True)
    #b2["label"] = b2["label"].apply(lambda x:legends[x])
    cmi_mdf = pd.melt(b2, id_vars=['label'], var_name=['breakpoints'])
    
    fig,ax = plt.subplots(2,1,figsize=(12,15))
    sns.boxplot(x="breakpoints", y="value", hue="label", data=cci_mdf,ax=ax[0]) 
    sns.boxplot(x="breakpoints", y="value", hue="label", data=cmi_mdf,ax=ax[1]) 
    ax[0].set_title("Contact Index")
    ax[1].set_title("Mobility Index")
    for a in ax:
        a.legend(title ="cluster",loc=1)
        a.set_ylim(0.1,1.01)
        a.set_yticks(np.arange(0.1, 1, step=0.1))
        a.set_ylabel("MIC Correlation")
    ax[0].set_xlabel("")
    plt.xlabel("Start of Breakpoints")
    plt.savefig(output_path+fname,dpi=300,bbox_inches='tight')
    return a2,b2

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



def get_ern(cases_df):
    df = pd.DataFrame()
    df.index = cases_df.index
    for s in cases_df.columns:
        temp=cases_df[s]
        temp[temp<0]=0 #special treatment for Texas
        r_df = covid19.r_covid(temp,n_samples = 50, r_window_size=1)
        df[s]=r_df["R_mean"]
    return df

def changepoint_ind(data,plot = False):
    o = data
    data = moving_average(data,7)
    data = np.array(data).reshape(len(data),1)
    prior_function = partial(const_prior, p=1/(len(data) + 1))
    Q, P, Pcp = offline_changepoint_detection(data, prior_function ,offline_ll.StudentT(),truncate=-40)
    psum = np.exp(Pcp).sum(0)

    changepoints = []
    candidate_ind = argrelextrema(psum, np.greater)
    for ind in candidate_ind[0]:
        if psum[ind]>0.1:
            changepoints.append(ind)

    ind = [1,8,12,16]
    b_dates = list(o.index[changepoints][ind])
    if plot:
        fig, ax = plt.subplots(2, figsize=[18, 16], sharex=True)
        ax[0].plot(data)
        ax[1].plot(psum)
        plt.xticks(ticks = np.arange(0,700,step=100),labels = o.index[np.arange(0,700,step=100)])
        for i in ind:
            x = changepoints[i]
            y = psum[x]
            txt = o.index[changepoints][i]
            ax[1].text(x-25,y+0.05,txt,fontsize=14)
            ax[1].scatter(x,y,c="red")
            ax[0].vlines(x=x,ymin= 0, ymax = max(data),colors='red', ls='--', lw=2)
        ax[1].set_ylim(-0.05,0.8)
        ax[1].set_xlabel("Date",labelpad=10)
        ax[1].set_ylabel("Probability of a Changepoint",labelpad=10)
        ax[0].set_ylabel("Number of Daily COVID Cases",labelpad=10)
        
        plt.savefig(output_path+"changepoint.pdf",dpi=300,bbox_inches='tight')
    return b_dates

#t is time series
#k specifies the number of clusters
def kshape_clust(t,k):
    #np.random.seed(71)
    time_series = t.transpose().to_numpy()
    # smooth = []
    # for x in time_series:
    #     smooth.append(moving_average(x,7))
    clusters = kshape(zscore(time_series,axis=1),k)
    mem = [clusters[i][1] for i in range(k)]
    kshape_labels = []
    for i in range(len(t.columns)):
        for j in range(k):
            if i in mem[j]:
                kshape_labels.append(j)
                break
    joined_labels=pd.concat([pd.Series(t.columns,name = "state"),pd.Series(kshape_labels,name="label")],axis=1)
    return joined_labels,clusters

def plot_centroids(centers,cp_index,smooth = True):
    cp_index = [a-7 for a in cp_index]
    fig,axes = plt.subplots(nrows=1,ncols=3,sharey = True, figsize = (25,6))
    for i in range(len(axes)):
        if smooth:
            x = moving_average(centers[i][0],7)
            height = 4
        else:
            x = centers[i][0]
            height = 10
        axes[i].plot(x,lw=2)
        axes[i].set_title(f"Cluster {i} centroid")
        axes[i].set_xlabel("Date")
        axes[i].vlines(x=cp_index,ymin= -1, ymax = height,colors='red', ls='--', lw=2)
        axes[i].set_xticks([100,350,600],["2020-07-17","2021-03-24","2021-11-29"])
    axes[0].set_ylabel("Z-Normalized COVID-19 Daily Cases")
    plt.savefig(output_path+"centroids.pdf",dpi=300,bbox_inches="tight")

def get_silhouette(max_k,n_iter,df,fname ="Silhouette.pdf"):
    np_df = df.transpose().to_numpy()
    silhouette = []
    for k in range(2,max_k+1):
        s = -1
        for i in range(n_iter):
            l,c=kshape_clust(df,k)
            sil_score = silhouette_score(zscore(np_df,axis=1),l["label"].to_numpy())
            s = max(s,sil_score)
        silhouette.append(s)
    plt.subplots(figsize = (8,5))
    plt.plot(range(2,max_k+1),silhouette)
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Curve")
    plt.savefig(output_path+fname,dpi=300,bbox_inches="tight")

##label needs to be a df with state as names
def plot_clust(label,k):
    fig, ax = plt.subplots(figsize=(12, 7))
    us = gpd.read_file("zip://../2021data/cb_2018_us_state_5m.zip")
    fil_func = lambda x: x in ["HI","GU","MP","AS","PR","AK","VI"]
    sub_us = us[pd.Series(not fil_func(a) for a in us["STUSPS"])]
    sub_us = sub_us.to_crs("EPSG:3395")
    partitioned_us = sub_us.merge(label,left_on = "NAME",right_on="state",how="left")
    color_mapping = {0: "tab:blue", 1: "darkorange", 2 :"green", 3: "firebrick"}
    partitioned_us.plot(color=partitioned_us["label"].map(color_mapping),ax=ax,edgecolor="black")
    custom_points = [Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=color) for color in list(color_mapping.values())[:k]]
    leg_points = ax.legend(custom_points, list(color_mapping.keys())[:k],loc=4)
    ax.add_artist(leg_points)
    #plt.title(f"US states with {k} clusters")
    plt.xticks([]),plt.yticks([])
    plt.savefig(output_path+f"cluster/{k}_cluster.pdf",dpi=300,bbox_inches='tight')

## group needs to be a list of state orders
def opt_time_lag(label_df,start_time,end_time,a,b,k):
    group = label_df["state"]
    cut = set()
    for i in range(k):
        cut.add(label_df[label_df["label"]==i].iloc[0]["state"])
    df = pd.DataFrame()
    df.index = a.columns
    df2 = pd.DataFrame()
    df2.index = a.columns
    for t in time_lags:
        col_mic = []
        col_spearman=[]
        for state in df.index:
            x = moving_average(a[state],t)
            y = moving_average(b[state],t)
            col_mic.append(get_mic(x,y))
            r, p = scipy.stats.spearmanr(x, y)
            col_spearman.append(r)
        df[t] = col_mic
        df2[t]=col_spearman
    ##df.reindex(group).style.background_gradient(cmap='Blues')
    df=df.reindex(group)
    df2=df2.reindex(group)
    sns.set(rc={"figure.figsize":(15,12)})
    plt.clf()
    s = sns.heatmap(df,cmap=cmap_color,yticklabels=True,xticklabels=True,annot=True,vmin=0.2,vmax=1)
    for tick_label in s.axes.get_yticklabels():
        if tick_label.get_text() in cut:
            tick_label.set_color("red")
    plt.title(start_time+" to "+end_time) 
    plt.suptitle("MIC correlation heatmap")
    name = output_path+f"{idx_type}-correlation/{start_time[5:]} to {end_time[5:]}_mic.pdf"
    plt.savefig(name) 
    plt.clf()
    s = sns.heatmap(df2,cmap=cmap_color,yticklabels=True,xticklabels=True,annot=True,vmin=-1,vmax=1)
    for tick_label in s.axes.get_yticklabels():
        if tick_label.get_text() in cut:
            tick_label.set_color("red")
    plt.title(start_time+" to "+end_time)
    plt.suptitle("Spearman correlation heatmap")
    name = output_path+f"{idx_type}-correlation/{start_time[5:]} to {end_time[5:]}_spearman.pdf"
    plt.savefig(name)

#changepoint
def main():
    parser = argparse.ArgumentParser(description='Covid Correlation Analysis')
    parser.add_argument("index",type=str)
    parser.add_argument("start_date",type=str)
    parser.add_argument("end_date",type = str)
    parser.add_argument("--k",type = int, default=2)
    parser.add_argument("--plot_clust", type=str, default=False)
    parser.add_argument("--plot_box", type=str, default=False)
    args = parser.parse_args()
    s = args.start_date
    e = args.end_date
    global idx_type
    idx_type = args.index
    global cmap_color
    if idx_type=="cci":
        idx = cci
        cmap_color="Reds"
    else:
        idx = cmi
        cmap_color = "Blues"
    idx2=idx[s:e].copy()
    daily2=daily[s:e].copy()
    #get_stats(idx2,daily2,False)
    lab,c = kshape_clust(daily,args.k) #always cluster based on cci
    lab = lab.sort_values(by = 'label')
    if args.plot_clust=="True":
        plot_clust(lab,args.k)
    opt_time_lag(lab,s,e,idx2,daily2,args.k)

if __name__=="__main__":
    main()