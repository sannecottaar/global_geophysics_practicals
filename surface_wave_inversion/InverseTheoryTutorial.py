# -*- coding: utf-8 -*-
import numpy as np
from scipy.sparse import coo_matrix
from inversetutorial_subs import blks2d, sray, make_sh_checkerboard, blks_resample, calc_dist, blks_latlon
import matplotlib.pyplot as plt
import rem3d
#from mpl_toolkits.basemap import Basemap

# Script to calculate G matrix for a surface wave dataset using ray theory

# BLOCK ONE: Set some basic variables for use in the script

#model and data setup
blocksize = float(input('Set an approximate block dimension in degrees = ')); 
#blocksize = 3.; #the approximate block dimension in degrees

dataperiod=input('Set a central period of the data in seconds (050, 100, or 150) = '); 
#dataperiod = 100; #the central period of the data in s (050,100, or 150)

# Specify the file containing phase measurements (2 provided with tutorial)
datafile= "GDM52.0.R1.%d.grid2562.homo.REM3D" % (dataperiod); #the phase measurement file
#datafile = "Scripps_Ma14.0.R1.%d.grid2562.homo.REM3D" % (dataperiod)

# Reference velocity for Rayleigh waves of chosen period? (100s Rylgh --> 4.088km/s, 100s Love --> 4.462)
cref=float(input('Set the reference velocity (km/s) appropriate for your choice of central period = ')); 
#cref = 4.088; 

# Do you want to increase / decrease reported error estimates on data?
emult=float(input('Set a multiplier on data error estimates = '));
#emult = 1.0; 

#%% BLOCK TWO: Compute sensitivity matrix (sparse and large) and get data vector and inverse data covariance matrix

# Create the pixel-like grid on which to parameterize the inverse problem
[nblk,bsize,nlat,mlat,hsize] = blks2d(blocksize)
print "There are %d blocks in the model" % (nblk);

# Read in the data from the file
A = np.loadtxt(datafile,dtype={'names':('overtone','peri','typeiorb','cmtname','eplat','eplon','cmtdep','stat','stlat','stlon','distkm','refphase','delobsphase','delerrphase','delpredphase'),'formats':('i','f','S','S','f','f','f','S','f','f','f','f','f','f','f')})
ndatamax=A.shape[0];

print "There are %d measurements in file %s\n" % (ndatamax,datafile);
  
# Set up space for some matrices and vectors
print "Calculating G matrix";
    
#some constants
rad=np.pi/180.0;
fac=2*np.pi*6371.0/360.0;
nproc = 0; nskip = 0; 

d_obs_temp=np.zeros(ndatamax);
d_err_temp=np.zeros(ndatamax);
      
for n in range(0,ndatamax):
    #norb   = A(4);
    norb = 1; 
    period = A[n][0];
    cmtlat = A[n][4];
    cmtlon = A[n][5];
    stlat  = A[n][8];
    stlon  = A[n][9];
    dT     = A[n][12];
    dTstd  = A[n][13];
                     
    t0=(90.0-cmtlat)*rad;
    p0=cmtlon*rad;
    #t0=geocen(t0);
            
    ts=(90.0-stlat)*rad;
    ps=stlon*rad;
    #ts=geocen(ts);
            
    #calculate row of G matrix according to distances from ray theory
    [row,delt]=sray(t0,p0,ts,ps,norb,nblk,nlat,bsize,mlat,hsize);
    # Reference travel-time for path
    dT0 = fac*delt/cref;
            
    # Row is path length in each box and delt is the total distance       
    rowsum=np.sum(row/delt);
    if(np.abs(rowsum-1.0)>0.005): #throw out data if total sum of row ~= delt
         nskip=nskip+1;
         print "skip sum for datum %d\n" % (n)
            
    if(np.mod(nproc,1000)==0): #output status every 100 measurements
         print "Working on path %d out of %d" % (nproc,ndatamax);
        
    #Add in row to sparse G matrix
    G_temp=0.01*row/delt; #conversion for model in percent perturbation
    d_obs_temp[nproc]=dT/dT0;
    d_err_temp[nproc]=dTstd/dT0;
    if(cmtlon<0.): 
        cmtlon = cmtlon+360.;
    if(stlon<0.): 
        stlon = stlon+360.;
   
    if(nproc==0):
        J = [G_temp.nonzero()];
        S = [G_temp.compress((G_temp!=0).flat)];
        sta_loc = [stlat,stlon];
        evt_loc = [cmtlat,cmtlon];
    else:
#        I.append(nproc*np.ones(np.shape(G_temp.nonzero()),dtype=np.int);
        J.append(G_temp.nonzero());
        S.append(G_temp.compress((G_temp!=0).flat)); 
        sta_loc.append([stlat,stlon]);
        evt_loc.append([cmtlat,cmtlon]);
 
    nproc=nproc+1;

# Because data error estimates vary a lot in our datasets, we approximate them
# by adding the median uncertainty to all estimates and dividing by 2    
d_err_temp = 0.5*(d_err_temp+np.median(d_err_temp));
    
# Compute square root of inverse data covariance matrix --> assume uncorrelated errors equal for each path
Cdinv = (emult*d_err_temp)**(-1);

# Construct sparse matrix relating model parameters to observations
G = coo_matrix((nproc,nblk),dtype=np.float64).toarray();
for n in range(nproc):
    G[n,J[n]]=Cdinv[n]*S[n];

#%% BLOCK THREE: Plot up data coverage (hit-count map)

# Compute data coverage (number of hits) in each model block
mhit=np.zeros(nblk);
for n in range(nblk):    
    mhit[n]=np.size(G[:,n].nonzero())
sampling = 1.;     

# Now, make a map of the data coverage contained in G    
[modlat,modlon,hz]=blks_resample(nblk,bsize,nlat,mlat,hsize,mhit,sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
hz = np.reshape(hz,np.size(hz));
latlonval = np.vstack((modlat,modlon,np.log10(hz))).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = latlonval[:,2];

# Plot hit-count map uisng rem3d routines
fig=plt.figure(figsize=(14,6)) 
ax=fig.add_subplot(1,1,1)
projection='robin'; vmin = 2.; vmax = 3.;
if projection=='ortho':
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='log(hitcount)',lat_0=0,lon_0=150,colorpalette='inferno')
else:
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='log(hitcount)',lat_0=0,lon_0=150,colorpalette='inferno')
ax.set_title('Hit-Count Map for '+str(dataperiod)+' sec Data')
plt.show()
#filename='hitcountmap'
#fig.savefig(filename+arg.format,dpi=300)


#plt.figure(figsize=(24,12))
#map = Basemap(projection='robin',lat_0=0, lon_0=180)
#map.drawcoastlines(color='lightgray')
#map.drawmapboundary()
#map.drawmeridians(np.arange(0, 360, 30))
#map.drawparallels(np.arange(-90, 90, 30))
#map.contourf(modlon, modlat, np.log10(hz),40,latlon=True); #, np.linspace(0,np.max(np.log10(hz)),50), cmap=plt.cm.gist_rainbow, origin='lower')
#plt.clim(2,3)
#plt.colorbar();
#plt.show();


#%% BLOCK FOUR: Perform Tarantola-Valette 1982 style inversion (prior information)
    
# Add prior information -- smoothing and/or damping
# Smoothing lengthscale (in degrees) --> if zero then only damping
# Make sure the smoothing doesn't correspond to the block size
smooth = float(input('Smoothing lengthscale (in degrees)? '));
smooth = smooth*rad; 

# Prior variance of model parameter distribution centered at zero. Good guess is between 1 and 100 (in percent)
varm = float(input('Prior variance on model parameters? '));

if(smooth==0):
    Cminv=(1/varm)*np.eye(nblk,nblk); #no covariance between model pars (i.e. no smoothing)
            
else:
    [blat,blon]=blks_latlon(nblk,bsize,nlat,mlat,hsize);
    delta=calc_dist(blat,blon);
    Cm=np.zeros((nblk,nblk));
    Cm[delta<30*smooth]=varm*np.exp(-0.5*(delta[delta<30*smooth]**2/(smooth**2)));
    U,S,V=np.linalg.svd(Cm);
    #Sp=S[S>0.0000001*S[0]];
    Sp = S; 
    p=np.size(Sp);
    Spinv = np.zeros((nblk,nblk));
    Spinv[0:p,0:p]=np.diag(1./Sp);
    Cminv= np.dot(U, np.dot(Spinv, V));    
    
GtG = np.matmul(np.transpose(G),G);
Gtd = np.matmul(np.transpose(G),Cdinv*d_obs_temp);
Cmpost = np.linalg.inv(GtG + Cminv); 
dg = np.matmul(Cmpost,Gtd); 

# Compute resolution matrix 
RmatTV = np.matmul(Cmpost,GtG);
print 'TV inversion resolved', np.int(np.trace(RmatTV)), 'model parameters'

# Resample the solution (dg) on a uniform grid for plotting
[modlat,modlon,dg_plot]=blks_resample(nblk,bsize,nlat,mlat,hsize,dg,sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
dg_plot = np.reshape(dg_plot,np.size(dg_plot));
latlonval = np.vstack((modlat,modlon,dg_plot)).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = -latlonval[:,2];

# Plot the uniform-grid sampled solution
fig=plt.figure(figsize=(12,6)) 
ax=fig.add_subplot(1,1,1)
projection='robin'; vmin = -5.; vmax = 5.;
if projection=='ortho':
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='$\Delta V / V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
else:
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='$\Delta V / V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
ax.set_title('$\Delta V / V$ map using TV1982 inversion of '+str(dataperiod)+' sec Data')
plt.show()

#%% BLOCK FIVE: Plot uncertainty estimate from Tarantola-Valette 1982 style inversion (prior information)

# Resample the diagonal of the posterior covariance matrix on a uniform grid for plotting
[modlat,modlon,cm_plot]=blks_resample(nblk,bsize,nlat,mlat,hsize,np.sqrt(np.diag(Cmpost)),sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
cm_plot = np.reshape(cm_plot,np.size(cm_plot));
latlonval = np.vstack((modlat,modlon,cm_plot)).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = latlonval[:,2];

# Plot the uniform-grid sampled uncertainty
fig=plt.figure(figsize=(12,6)) 
ax=fig.add_subplot(1,1,1)
projection='robin'; vmin = 0.; vmax = np.max(cm_plot);
if projection=='ortho':
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='$1\sigma$ Uncertainty',lat_0=0,lon_0=150,colorpalette='inferno')
else:
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='$1\sigma$ Uncertainty',lat_0=0,lon_0=150,colorpalette='inferno')
ax.set_title('1SD uncertainty (from posterior Cm) in TV1982 inversion of '+str(dataperiod)+' sec Data')
plt.show()


#%% BLOCK SIX: Carry out a checkerboard test to assess resolution of TV1982 inversion

# What spherical harmonic equivalent is the checkerboard characteristic length scale?
chk_l = int(input('Spherical harmonic degree of checkerboard? '));

ckmodel=make_sh_checkerboard(chk_l,nblk,bsize,nlat,mlat,hsize);
ckmodel_out=np.matmul(RmatTV,ckmodel);

# Resample the input and output checkerboards on a uniform grid for plotting
[modlat,modlon,ck_in_plot]=blks_resample(nblk,bsize,nlat,mlat,hsize,ckmodel,sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
ck_in_plot = np.reshape(ck_in_plot,np.size(ck_in_plot));
latlonval = np.vstack((modlat,modlon,ck_in_plot)).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = latlonval[:,2];

# Plot the input checkerboard
fig=plt.figure(figsize=(24,12)) 
ax=fig.add_subplot(2,1,1)
projection='robin'; vmin = 1.3*np.min(ckmodel); vmax = 1.3*np.max(ckmodel);
if projection=='ortho':
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
else:
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
ax.set_title('Input Checkerboard for TV1982 inversion of '+str(dataperiod)+' sec Data')

# output pattern
[modlat,modlon,ck_out_plot]=blks_resample(nblk,bsize,nlat,mlat,hsize,ckmodel_out,sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
ck_out_plot = np.reshape(ck_out_plot,np.size(ck_out_plot));
latlonval = np.vstack((modlat,modlon,ck_out_plot)).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = latlonval[:,2];

# Plot the output checkerboard
ax2=fig.add_subplot(2,1,2)
projection='robin'; vmin = 1.3*np.min(ckmodel); vmax = 1.3*np.max(ckmodel);
if projection=='ortho':
    rem3d.plots.globalmap(ax2,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
else:
    rem3d.plots.globalmap(ax2,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
ax2.set_title('Output Checkerboard for TV1982 inversion of '+str(dataperiod)+' sec Data')

plt.show()


#%% BLOCK SEVEN: Invert using SVD -- no need for prior information 

# How many model parameters to estimate? A good starting number is int(nblk/2)
pp = int(input('How many model parameters to estimate? '));

U,S,V = np.linalg.svd(GtG); 
Sp = S[0:pp-1];
Spinv = np.zeros((nblk,nblk));
Spinv[0:pp-1,0:pp-1]=np.diag(1./Sp);
GtGinv= np.dot(U, np.dot(Spinv, V));    

RmatSVD = np.matmul(GtGinv,GtG);
print 'SVD inversion resolved', np.int(np.trace(RmatSVD)), 'model parameters'

dg2 = np.matmul(GtGinv,Gtd); 
[modlat,modlon,dg_plot]=blks_resample(nblk,bsize,nlat,mlat,hsize,dg2,sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
dg_plot = np.reshape(dg_plot,np.size(dg_plot));
latlonval = np.vstack((modlat,modlon,dg_plot)).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = -latlonval[:,2];

fig=plt.figure(figsize=(24,12)) 
ax=fig.add_subplot(1,1,1)
projection='robin'; vmin = -5.; vmax = 5.;
if projection=='ortho':
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='$\Delta V / V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
else:
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='$\Delta V / V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
ax.set_title('$\Delta V / V$ map using SVD inversion of '+str(dataperiod)+' sec Data')
plt.show()

#%% BLOCK EIGHT: Carry out a checkerboard test to assess resolution of SVD inversion

# What spherical harmonic equivalent is the checkerboard characteristic length scale?
chk_l = int(input('Spherical harmonic degree of checkerboard? '));

ckmodel=make_sh_checkerboard(chk_l,nblk,bsize,nlat,mlat,hsize);
ckmodel_out=np.matmul(RmatSVD,ckmodel);

# Resample the input and output checkerboards on a uniform grid for plotting
[modlat,modlon,ck_in_plot]=blks_resample(nblk,bsize,nlat,mlat,hsize,ckmodel,sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
ck_in_plot = np.reshape(ck_in_plot,np.size(ck_in_plot));
latlonval = np.vstack((modlat,modlon,ck_in_plot)).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = latlonval[:,2];

# Plot the input checkerboard
fig=plt.figure(figsize=(24,12)) 
ax=fig.add_subplot(2,1,1)
projection='robin'; vmin = 1.3*np.min(ckmodel); vmax = 1.3*np.max(ckmodel);
if projection=='ortho':
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
else:
    rem3d.plots.globalmap(ax,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
ax.set_title('Input Checkerboard for SVD inversion of '+str(dataperiod)+' sec Data')

# output pattern
[modlat,modlon,ck_out_plot]=blks_resample(nblk,bsize,nlat,mlat,hsize,ckmodel_out,sampling); #sample model on regular grid
modlat = np.reshape(modlat,np.size(modlat));
modlon = np.reshape(modlon,np.size(modlon));
ck_out_plot = np.reshape(ck_out_plot,np.size(ck_out_plot));
latlonval = np.vstack((modlat,modlon,ck_out_plot)).transpose()
dt = {'names':['lat', 'lon', 'val'], 'formats':[np.float, np.float, np.float]}
plotmodel = np.zeros(len(latlonval), dtype=dt);
plotmodel['lat'] = latlonval[:,0]; 
plotmodel['lon'] = latlonval[:,1]; 
plotmodel['val'] = latlonval[:,2];

# Plot the output checkerboard
ax2=fig.add_subplot(2,1,2)
projection='robin'; vmin = 1.3*np.min(ckmodel); vmax = 1.3*np.max(ckmodel);
if projection=='ortho':
    rem3d.plots.globalmap(ax2,plotmodel,vmin,vmax,grid=[30.,30.],gridwidth=1,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
else:
    rem3d.plots.globalmap(ax2,plotmodel,vmin,vmax,grid=[30.,90.],gridwidth=0,projection=projection,colorlabel='$\Delta V/V$ (%)',lat_0=0,lon_0=150,colorpalette='rem3d')
ax2.set_title('Output Checkerboard for SVD inversion of '+str(dataperiod)+' sec Data')

plt.show()


