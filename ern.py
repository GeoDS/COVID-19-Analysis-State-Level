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
import epyestim
import epyestim.covid19 as covid19

path="../2021data/"
cci=pd.read_csv(path+'2021_state_cci.csv',index_col = "date")
cmi=pd.read_csv(path+'2021_state_cmi.csv',index_col = "date")
# cases=pd.read_csv(path+"2021_state_cases.csv",index_col="date")
# deaths=pd.read_csv(path+"2021_state_deaths.csv",index_col="date")
# daily = pd.read_csv(path+"2021_daily_cases.csv",parse_dates=["date"],index_col="date")
ern = pd.read_csv(path+"ern.csv",index_col="date")
time_lags = [3,6,9,12,15,18,21]


def get_mic(a,b):
    mine=MINE(est="mic_approx")
    mine.compute_score(a,b)
    return mine.mic()


def get_stats(a,b,box=False):
    mic = []
    spearman = []
    for state in a.columns:
        x = a[state]
        y = b[state]
        m = get_mic(x,y)
        r, p = spearmanr(x, y)
        mic.append(m)
        spearman.append(r)
    if box: 
        plt.boxplot(mic)
        plt.title("MIC Correlation")
        plt.show()

        plt.boxplot(spearman)
        plt.title("Spearman's Correlation")
        plt.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# length becomes 350 needs to investigate
def get_ern(cases_df):
    df = pd.DataFrame()
    df.index = cases_df.index[6:-10]
    for s in cases_df.columns:
        temp=cases_df[s]
        temp[temp<0]=0 #special treatment for Texas
        r_df = covid19.r_covid(temp,r_window_size=1)
        df[s]=r_df["R_mean"]
    return df

#t is time series
#k specifies the number of clusters
def kshape_clust(t,k):
    np.random.seed(26)
    time_series = []
    for f in t.columns:
        time_series.append(list(t[f]))
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

##label needs to be a df with state as names
def plot_clust(label):
    us = gpd.read_file("zip://../2021data/cb_2018_us_state_5m.zip")
    fil_func = lambda x: x in ["HI","GU","MP","AS","PR","AK","VI"]
    sub_us = us[pd.Series(not fil_func(a) for a in us["STUSPS"])]
    sub_us = sub_us.to_crs("EPSG:3395")
    partitioned_us = sub_us.merge(label,left_on = "NAME",right_on="state",how="left")
    color_mapping = {0: "tab:blue", 1: "lightgreen", 2 :"#fb8072", 3: "#beaed4"}
    partitioned_us.plot(color=partitioned_us["label"].map(color_mapping))
    plt.show()

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
    s = sns.heatmap(df,cmap=cmap_color,yticklabels=True,xticklabels=True,annot=True,vmin=0.2,vmax=1)
    for tick_label in s.axes.get_yticklabels():
        if tick_label.get_text() in cut:
            tick_label.set_color("red")
    plt.title(start_time+" to "+end_time) 
    plt.suptitle("MIC correlation heatmap")
    name = f"../results/ern/{idx_type}-mic-{k}.pdf"#f"../results/{idx_type}-correlation/{start_time[5:]} to {end_time[5:]}_mic.pdf"
    plt.savefig(name) 
    plt.clf()
    s = sns.heatmap(df2,cmap=cmap_color,yticklabels=True,xticklabels=True,annot=True,vmin=-1,vmax=1)
    for tick_label in s.axes.get_yticklabels():
        if tick_label.get_text() in cut:
            tick_label.set_color("red")
    plt.title(start_time+" to "+end_time)
    plt.suptitle("Spearman correlation heatmap")
    name = f"../results/ern/{idx_type}-spearman-{k}.pdf"#f"../results/{idx_type}-correlation/{start_time[5:]} to {end_time[5:]}_spearman.pdf"
    plt.savefig(name)

#changepoint 05-23 08-05
def main():
    parser = argparse.ArgumentParser(description='Covid Correlation Analysis')
    parser.add_argument("index",type=str)
    parser.add_argument("start_date",type=str)
    parser.add_argument("end_date",type = str)
    parser.add_argument("k",type = int)
    parser.add_argument("plot_clust", type=str)
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
    ern2 = ern[s:e].copy()
    get_stats(idx2,ern,False)
    lab,c = kshape_clust(cci,args.k) #always cluster based on cci
    lab = lab.sort_values(by = 'label')
    if args.plot_clust=="True":
        plot_clust(lab)
    opt_time_lag(lab,s,e,idx2,ern2,args.k)

if __name__=="__main__":
    main()