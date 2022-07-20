import pandas as pd
import numpy as np
from scipy import stats
from brainsmash.mapgen.base import Base 
from brainsmash.mapgen.eval import base_fit
from brainsmash.mapgen.stats import pearsonr, pairwise_r, nonparp


def metric2mmp(df,sel_met,roi_id,median=True,hemi='L',calc_log=False):
    avg_vox_vals_mmp = df.copy() #slope_per_roi
    if median:
        avg_vox_vals_mmp = avg_vox_vals_mmp.groupby(roi_id, as_index=False).mean()
    else:
        agg_mode_text = {sel_met: stats.mode}
        avg_vox_vals_mmp = avg_vox_vals_mmp.groupby(roi_id, as_index=False).agg(agg_mode_text)
        avg_vox_vals_mmp[sel_met] = avg_vox_vals_mmp[sel_met].str[0].str[0]
    last_roi = 181 if hemi=='L' else 361
    sel_rois = ((avg_vox_vals_mmp[roi_id]>0) & (avg_vox_vals_mmp[roi_id]<181)) if hemi=='L' else (avg_vox_vals_mmp[roi_id]>180)   
    avg_vox_vals_mmp = avg_vox_vals_mmp.loc[sel_rois]
    missing_rois = list(avg_vox_vals_mmp[roi_id].unique().astype(int))
    last_roi_missing=True if (missing_rois[-1]!=last_roi) else False
    missing_rois = sorted(set(range(1, last_roi)) - set(missing_rois)) if hemi=='L' else sorted(set(range(181, last_roi)) - set(missing_rois))
    if last_roi_missing: missing_rois+=[180]
    print('Missing ROIs: {}'.format(missing_rois))
    min_val = avg_vox_vals_mmp[sel_met].min()
    min_val = min_val-1 if min_val <0 else 0 #
    for rid in missing_rois:
        avg_vox_vals_mmp = avg_vox_vals_mmp.append({roi_id:rid,sel_met:min_val}, ignore_index=True)
    avg_vox_vals_mmp = avg_vox_vals_mmp.sort_values(by=roi_id)
    mmp_sel_met= avg_vox_vals_mmp[sel_met].to_numpy()
    if calc_log: mmp_sel_met = np.log(mmp_sel_met)
    mmp_sel_met[np.isnan(avg_vox_vals_mmp[sel_met].to_numpy())]=min_val if not calc_log else np.log(min_val)
    return mmp_sel_met

def valid_data_index(x,y,n_mad=2):
    x[pd.isna(x)]=x.min()
    y[pd.isna(y)]=y.min()
    x = x.astype(float)
    y = y.astype(float)
    if not isinstance(n_mad, str):
        x_mad = stats.median_absolute_deviation(x[x>x.min()],nan_policy='omit')
        x_med = np.nanmedian(x[x>x.min()])
        y_mad = stats.median_absolute_deviation(y[y>y.min()],nan_policy='omit')
        y_med = np.nanmedian(y[y>y.min()])
        valid_ind = ((x<(x_med+(n_mad*x_mad))) & (x>(x_med-(n_mad*x_mad))) & (y<(y_med+(n_mad*y_mad))) & (y>(y_med-(n_mad*y_mad))) & (x>x.min()) & (y>y.min()))
    else:
        valid_ind = ((x>np.nanmin(x)) & (y>np.nanmin(y)))
    return valid_ind

def smash_comp(x,y,distmat,y_nii_fn='',xlabel='x',ylabel='y',cmap='summer',n_mad=2,rnd_method='smash',l=5,u=95,p_uthr=0.06,colorbar=True,xlim=None,ylim=None,p_xlim=[-0.5,0.5],plot=True,print_text=False,plot_rnd=True,plot_surface=True,x_surr_corrs=None):
    valid_ind = valid_data_index(x,y,n_mad=n_mad)
    test_r,test_p = stats.pearsonr(x[valid_ind], y[valid_ind])
    if test_p<p_uthr:
        dist = distmat[valid_ind,:]
        dist = dist[:,valid_ind]
        if (x_surr_corrs is None):
            if rnd_method=='smash':
                x_gen = Base(x[valid_ind], dist)  # note: can pass numpy arrays as well as filenames
                x_surr_maps = x_gen(n=1000)           
            else:
                x_surr_maps = np.array([np.random.permutation(x[valid_ind]) for _ in range(1000)])
            x_surr_corrs = pearsonr(y[valid_ind], x_surr_maps).flatten()
            return_x_surr_corrs = True
        else:
            return_x_surr_corrs = False
        p_np = nonparp(test_r, x_surr_corrs)
        if plot:
            if p_np<0.08:
                plot_joint(x[valid_ind],y[valid_ind],s=28,robust=False,x_label=xlabel,y_label=ylabel,xlim=xlim,ylim=ylim,p_smash=p_np)
            else:
                plot_joint(x[valid_ind],y[valid_ind],s=28,robust=False,x_label=xlabel,y_label=ylabel,xlim=xlim,ylim=ylim)
            if plot_rnd:
                plt.figure(figsize=(2,5))
                plot_rnd_dist(x_surr_corrs,test_r,p_np,plt.gca(),xlabel=xlabel,ylabel=ylabel,xlim=p_xlim,print_text=print_text)
            if plot_surface: plot_surf(y,remove_ext(y_nii_fn)+'_LH_ROIwise',vlow=l,vhigh=u,cmap=cmap,colorbar=colorbar)
        if return_x_surr_corrs:
            return x_surr_corrs
    else:
        return np.array([])

def vrange(x,l=5,u=95):
    return (np.nanpercentile(x, l), np.nanpercentile(x,u))