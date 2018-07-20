# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:19:19 2018
This script contains all subroutines needed by the CIDER tutorial 
@author: Ved
"""

#####################  IMPORT STANDARD MODULES   ############################## 
import numpy as np


#####################  SUBROUTINES USED FOR SURFACE WAVE TOMOGRAPHY ###########
def shfcn(l,c,s):
    #%SHFCN COmpute ordinary spherical harmonics
    #%  computes ordinary spherical harmonics (Edmonds eqn 2.5.29)
    #%    x(l,m,theta)*exp(imphi) are a set of orthonormal spherical 
    #%    harmonics where theta is colatitude and phi is longitude.
    #%  input:
    #%    l=harmonic degree
    #%    c=cos(theta)
    #%    s=sin(theta) 
    #%  output:
    #%    x(1) contains m=0, x(2) contains m=1, ... x(l+1) contains m=l
    #%    where m=azimuthal order 0.le.m.le.l .  
    #%    dxdth (theta derivatives) stored in the same way as x
    #%  calls no other routines
    #%       TGM
    #%  translated quickly to matlab from Guy Master's fortran and then to Python
    lp1=l+1;
    fl2p1=l+lp1;
    con=np.sqrt(fl2p1/(4*np.pi));
    x=np.zeros(int(lp1));
    dxdth=np.zeros(int(lp1));
      
    #%*** handle special case of l=0
    if(l==0):
        x[0]=con;
        dxdth[0]=0.;
        return (x,dxdth)
      
    if(np.abs(s)<1e-20):
    #%*** handle very small arguments
        x[0]=con;
        dxdth[1]=-0.5*con*np.sqrt(l*lp1);
        return (x,dxdth)

    #%*** try m recursions ; first compute xll
    f=1.0;
    for i in range(1,int(lp1)):
        f=f*(i+i-1.)/(i+i);
      
    if(np.log(s)*l>-679):
    #%*** use m recurrence starting from m=l
        cot=c/s;                      
        x[l]=con*np.sqrt(f)*(-s)**float(l);
        dxdth[l]=x[l]*l*cot;
        for i in range(1,int(lp1)):
            m=lp1-i;
            mp1=m+1;   
            f=np.sqrt(float(i)*(fl2p1-i));
            x[int(m)-1]=-(dxdth[int(mp1)-1]+float(m)*cot*x[int(mp1)-1])/f;
            dxdth[int(m)-1]=(m-1)*x[int(m)-1]*cot+x[int(mp1)-1]*f;
    else:
        #%*** use the Libbrecht algorithm
        c2=c+c;
        x[int(lp1)-1]=con*np.sqrt(f)*(-1.)**float(l);
        dxdth[int(lp1)-1]=0.;
        for i in range(1,int(lp1)):
            m=lp1-i;
            mp1=m+1;    
            f=np.sqrt(float(i)*(fl2p1-i));
            x[int(m)-1]=-(s*dxdth[int(mp1)-1]+float(m)*c2*x[int(mp1)-1])/f;
            dxdth[int(m)-1]=s*x[int(mp1)-1]*f;
        
#%*** now convert back to ordinary spherical harmonics
        fac=1.0;
        for i in range(2,int(lp1+1)):
            dxdth[i-1]=(s*dxdth[i-1]+float(i-1)*c*x[i-1])*fac;
            x[i-1]=x[i-1]*s*fac;
            fac=fac*s;

    return (x,dxdth)

def make_sh_checkerboard(l,nblk,bsize,nlat,mlat,hsize):
    #%MAKE_SH_CHECKERBOARD Makes a spherical harmonic checkerboard of order l
    #%   m is set to 1/2 l to give an approximate spherical harmonic derived
    #%   checkerboard
    convert=np.pi/180.;

    m=np.round(0.5*l);

    ckmodel=np.zeros(nblk);
    for i in range(nblk):
        [th,ph]=iblk(i,bsize,nlat,mlat,hsize);
        c=np.cos(th*convert);
        s=np.sin(th*convert);
        [x,dx]=shfcn(l,c,s);
        argm=m*ph*convert;
        #    %frp=x(m+1)*cos(argm);
        fip=x[int(m)]*np.sin(argm);
        if(fip>0):
            ckmodel[i]=10;            
        else:
            ckmodel[i]=-10;
    
    return ckmodel

def iblk( blockindx,bsize,nlat,mlat,hsize ):
    #%IBLK Returns theta, phi of center of block with index blockindx

    for i in range(nlat):
        j1=mlat[i]+1;
        j2=mlat[i+1];
        
        if(blockindx>=j1 and blockindx<=j2):
            th=(float(i)+0.5)*bsize;
            ph=(blockindx-float(j1)+0.5)*hsize[i];
            return (th,ph)
    th=-999.; ph=-999.;
    return (th,ph)

def tptocart( theta, phi ): 
    #%TPTOCART Converts theta phi coordinates on a unit sphere to Cartesian
    #%   Simple spherical to Cartesian conversion    
    
    s=np.sin(theta);
    cart = np.array([s*np.cos(phi), s*np.sin(phi), np.cos(theta)])
    return cart

def cart2sph(x,y,z):
    # CART2SPH Transform Cartesian to spherical coordinates.
    #   [TH,PHI,R] = CART2SPH(X,Y,Z) transforms corresponding elements of
    #   data stored in Cartesian coordinates X,Y,Z to spherical
    #   coordinates (azimuth TH, elevation PHI, and radius R).  The arrays
    #   X,Y, and Z must be the same size (or any of them can be scalar).
    #   TH and PHI are returned in radians.
    #
    #   TH is the counterclockwise angle in the xy plane measured from the
    #   positive x axis.  PHI is the elevation angle from the xy plane.
        
    hypotxy = np.hypot(x,y);
    #r = np.hypot(hypotxy,z);
    elev = np.arctan2(z,hypotxy);
    az = np.arctan2(y,x);

    return ( az,elev )

    
def euler_mp(sth,sph,rth,rph):

    #%EULER Finds Euler angles for a coordinate axis rotation
    #%   Finds the Euler angles alpha,beta,gamma that rotate the coordinate axes 
    #%   so that the z-axis is at the pole of the source-receiver great circle 
    #%   (s x r), and the x-axis is at the source. See Edmonds' Angular Momentum 
    #%   in Quantum Mechanics, page 7 for the angle conventions.
    #%     input: sth,sph = source coordinates in radians
    #%            rth,rph = receiver coordinates in radians
    #%     output: alpha,beta,gamma = euler angles in radians which rotate the 
    #%             original coordinate system to the one with the source-receiver 
    #%             great circle on the equator, the source at (PI/2,0). The minor 
    #%             arc to the receiver is in the positive phi direction.
    #%            del = source-receiver separation in radians.
    #%            pth,pph = source-receiver great circle pole location.

    #   % Get cartesian coordinates for source and receiver
    scart=tptocart(sth,sph);
    rcart=tptocart(rth,rph);
    delta=np.arccos(np.dot(scart,rcart));
    pcart=np.cross(scart,rcart);
    pth=np.arctan2(np.sqrt(pcart[0]**2+pcart[1]**2),pcart[2]);
    if(pcart[0]==0 and pcart[1]==0): 
        # special case of pole at z or -z
        pph=0.0;
    else:
        pph=np.arctan2(pcart[1],pcart[0]);
    
    alpha=pph;
    beta=pth;
    #%  the x'' axis (call it t) is at pth+pi/2,pph
    ttheta=pth + np.pi/2.0;
    tcart=tptocart(ttheta,pph);
    #%  the third Euler angle, gamma, rotates x'' to the source s.
    gamma=np.arccos(np.dot(scart,tcart));
    #%  form q = x'' x s to check the sign of gamma (q/|q| = +-p/|p|)
    qcart=np.cross(tcart,scart);
    sgn=np.dot(pcart,qcart);
    if(sgn<0.0): 
        gamma=-gamma;
    
    return( alpha,beta,gamma,delta ) 

def blks_resample(nblk,bsize,nlat,mlat,hsize,mest,sampling):
# BLKS_RESAMPLE Returns lat lon vectors for a model parameterization
#  input values nblk,bsize,nlat,mlat,hsize are return values from
#  blks2d.  output is resampled to have even bsize spacing in both latitude
#  and longitude

   nlat_sample=np.int(np.round(180/sampling));
   sampsize=180./nlat_sample;

   nlon=2*nlat_sample;
   colat=np.transpose(np.tile(np.arange(0.5*sampsize,180.,sampsize),(nlon,1)));
   lat=90.0-colat;
   lon=np.tile(np.arange(0.5*sampsize,360.,sampsize),(nlat_sample,1));
   indx=np.round(fblk(colat,lon,nlat,bsize,mlat,hsize));  
   mz=mest[indx.astype(int)];

   return ( lat,lon,mz )

def blks_latlon(nblk,bsize,nlat,mlat,hsize):
   #BLKS_LATLON Returns lat lon vectors for a model parameterization
   #  input values nblk,bsize,nlat,mlat,hsize are return values from
   #  blks2d

   lat=np.zeros(nblk);
   lon=np.zeros(nblk);

   ib=0;
   for i in range(nlat):
       lati=90.0-((float(i)+0.5)*bsize);
       jjj = mlat[i+1]-mlat[i];
       for jj in range(int(jjj)):
            lonj=(float(jj)+0.5)*hsize[i];
            lat[ib]=lati;
            lon[ib]=lonj;
            ib=ib+1;

   return ( lat,lon )


def calc_dist( blat,blon ):
   # computes the matrix of distances

   dim=np.size(blat,axis=0);
   delta=np.zeros((dim,dim));
   # Convert to radians for trigonometric functions
   blat = np.pi/180.*blat; blon = np.pi/180.*blon; 
   for i in range(dim-1):
       delta[i,i+1:dim]=np.arccos(np.sin(blat[i])*np.sin(blat[i+1:dim]) + np.cos(blat[i])*np.cos(blat[i+1:dim])*np.cos(blon[i+1:dim]-blon[i]));
   delta=delta+np.transpose(delta); #fill in whole matrix

   return (delta) 


def blks2d(blocksz):
    #BLKS2D Establishes block parameters
    #  Divides a spherical surface into approximately blocksz x blocksz blocks
    #  where blocksz is degrees measured at the equator
    #  input
    #    blocksz - the desired block size (in degrees)
    #  outputs
    #    nblk - total number of blocks in the model
    #    bsize - the nominal output block size in degrees (may differ to be a
    #            factor of 180)
    #    nlat - the number of latitudinal samples
    #    mlat - the block index limits for each latitudinal band of blocks
    #    hsize - the dimension of each block in longitudinal degrees for each
    #            latitudinal band of blocks
 
   rconv=180./np.pi;
    
   nlat=np.int(180./blocksz);
   bsize=180./nlat;
    
   ib=0;
   tmax=0.0;
   mlat=np.zeros(nlat+1);
   hsize=np.zeros(nlat);
   for i in range(0,nlat):
        
       tmin=tmax;
       tmax=tmin+bsize;
       th=0.5*(tmin+tmax)/rconv;
       s1=np.sin(th);
       mlon=np.maximum(np.round((360./bsize)*s1),1);
       hsize[i]=360./mlon;
       ib=ib+mlon;
       mlat[i+1]=ib;
    
   nblk=np.int(ib);
   return ( nblk,bsize,nlat,mlat,hsize )
  
def fblk(t,p,nlat,bsize,mlat,hsize): 
   #FBLK Returns the index of block containing point (t,p) using grid info
   #   output from blks2d (nlat,bsize,mlat,hsize)
   #   t and p are permitted to be vectors

    it=np.floor_divide(t,bsize);
    it.clip(min=0, max=nlat);        
    j1=mlat[it.astype(int)];
    indx=np.floor_divide(p,hsize[it.astype(int)])+j1;
    return ( indx.astype(int) )    

def sray( sth,sph,rth,rph,norb,nblk,nlat,bsize,mlat,hsize ):
   #SRAY Determine a row of G matrix according to surface wave ray
   # theory

   row=np.zeros(nblk);
   rad=np.pi/180.0;
   pi2=0.5*np.pi;
    
   #Determine Euler angles to rotate source and receiver to equator with
   #source at x axis
   if(norb<10):
        [alpha,beta,gamma,delta]=euler_mp(sth,sph,rth,rph);
        alpha = np.real(alpha); beta  = np.real(beta);   # Added by Ved 12/2
        gamma = np.real(gamma); delta = np.real(delta); # Added by Ved 12/2
   else:
        beta=sth;
        alpha=sph;
        gamma=0.;
    
   calf=np.cos(alpha);
   salf=np.sin(alpha);
   cbet=np.cos(beta);
   sbet=np.sin(beta);
   cgam=np.cos(gamma);
   sgam=np.sin(gamma);
   if(norb>10):
        delta=2.0*np.pi;
   else:
        if(np.mod(norb,2)==1):
            delta=delta+(norb-1)*np.pi;
        else:
            delta=delta-norb*np.pi;
        
   delt=np.abs(delta)/rad;
   #*** note that delta is negative for even norb
   dp=0.01*rad;
   if(delta<0.0):
       dp=-dp;
    
   delp=np.abs(dp);
   #*** move around equator with small increments and add incremental path
   #lengths to appropriate block index for row in G matrix

   ps = np.arange(dp,delta,dp)
   x=np.cos(ps);
   y=np.sin(ps);
   t1=cgam*x-sgam*y;
   t2=sgam*x+cgam*y;
   u1=cbet*t1;
   z=-sbet*t1;
   x=calf*u1-salf*t2;
   y=salf*u1+calf*t2;
   [az,el]=cart2sph(x,y,z);
   az[az<0.] = az[az<0.]+2.0*np.pi;
   t0 = (pi2-el)/rad;
   p0 = az/rad; 
   p0[p0>=360.] = p0[p0>=360.]-360.; # If reach 360, wrap around
   ib=fblk(t0,p0,nlat,bsize,mlat,hsize);
   ib_unq,unq_count = np.unique(ib,return_counts=True);
   row[ib_unq.astype(int)] = delp/rad*unq_count;    
#   for jj in range(0,ib.size):      
#       row[ib[jj]]=row[ib[jj]]+delp/rad;
       
            
   #Do last step to delta if necessary
   if(np.abs(ps[-1])<np.abs(delta)):
        pslast=delta;
        delp=np.abs(pslast-ps[-1]);
        xlast=np.cos(pslast);
        ylast=np.sin(pslast);
        t1last=cgam*xlast-sgam*ylast;
        t2last=sgam*xlast+cgam*ylast;
        u1last=cbet*t1last;
        zlast=-sbet*t1last; 
        xlast=calf*u1last-salf*t2last;
        ylast=salf*u1last+calf*t2last;
        [azlast,ellast]=cart2sph(xlast,ylast,zlast);
        if(azlast<0.):
            azlast=azlast+2.0*np.pi;

        t0last=(pi2-ellast)/rad;
        p0last=azlast/rad; 
        if(p0last>=360.): 
            p0last=p0last-360; # If reach 360, wrap around
        
        iblast=fblk(t0last,p0last,nlat,bsize,mlat,hsize);
        row[iblast]=row[iblast]+delp/rad;
   
   return (row,delt)