#!/usr/bin/env python2.7

from __future__ import division
import med_trend_est as MTE
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

slope=0.2
nf=100
bf=80
step=3
per=365/step
time=sp.arange(0,1500,step)
dat=sp.zeros([len(time),5])
noise=nf*sp.rand(len(time))-(nf/2)
snoise=sp.stats.skewnorm.rvs(10,size=len(time))
dat[:,0]=slope*time+noise
dat[:,1]=dat[:,0]+50*sp.sin(2*sp.pi/365*time)
dat[:,2]=sp.hstack((dat[:len(time)//3,0],dat[len(time)//3:2*len(time)//3,0]+bf,dat[2*len(time)//3:len(time),0]+(2*bf)))
dat[:,3]=slope*time+sp.hstack((0.5*noise[:len(time)//3],noise[len(time)//3:2*len(time)//3],2*noise[2*len(time)//3:len(time)]))
dat[:,4]=slope*time+nf*snoise-sp.median(nf*snoise)
lab=('Linear+Noise','Linear+Noise+Annual','Linear+Noise+Steps','Linear+Noise+Heteroscedasticity','Linear+SkewedNoise')

k=0

indat=sp.column_stack((time,dat[:,k]))

A=sp.vstack([time,sp.ones(len(time))]).T
LSout=sp.linalg.lstsq(A,indat[:,1])[0]
LSerr=sp.real(sp.sqrt(sp.sum(sp.power(indat[:,1]-LSout[0]*time,2))/(len(time)-2)/sp.sum(sp.power(time-sp.mean(time),2))))
#LSerr=sp.std(indat[:,1]-LSout[0]*time)/len(time)


B=sp.vstack([time,sp.zeros(len(time))]).T
LS0out=sp.linalg.lstsq(B,indat[:,1])[0]
LS0err=sp.real(sp.sqrt(sp.sum(sp.power(indat[:,1]-LS0out[0]*time,2))/(len(time)-2)/sp.sum(sp.power(time-sp.mean(time),2))))
#LS0err=sp.std(indat[:,1]-LS0out[0]*time)/sp.sqrt(sp.sum(sp.power((time-sp.mean(time)),2)))

TSout=MTE.main([indat,'-TS','-h',lab[k]+'.TS'])
TSIAout=MTE.main([indat,'-TSIA','-h',lab[k]+'.TSIA','-per',str(per),'-int','N','-tol','10'])
MIDASout=MTE.main([indat,'-MIDAS','-h',lab[k]+'.MIDAS','-per',str(per),'-tol','10'])

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
DAT=ax.plot(time,indat[:,1],'x-',markersize=1,linewidth=0.5,color=(0.6,0.6,0.6))
TRUTH=ax.plot(time,slope*time,'-',color='k')
LS=ax.plot(time,LSout[0]*time+LSout[1],'-',color=(0.8,0,0))
LS0=ax.plot(time,LS0out[0]*time+LS0out[1],'--',color=(0.8,0,0))
TS=ax.plot(time,TSout[0]*time,'-',color='m')
TSIA=ax.plot(time,TSIAout[0]*time,'-',color='b')
MIDAS=ax.plot(time,MIDASout[0]*time,'-',color='g')
leg=ax.legend([DAT[0],LS[0],LS0[0],TS[0],TSIA[0],MIDAS[0],TRUTH[0]],
           ['Data      $\hat{m}$        $s_{\hat{m}}$',
           'LS:       '+str('%.2f' % LSout[0])+'  '+str('%.3f' % LSerr),
           'LSNI:    '+str('%.2f' % LS0out[0])+'  '+str('%.3f' % LS0err),
           'TS:       '+str('%.2f' % TSout[0])+'  '+str('%.3f' % TSout[3]),
           'TSIA:    '+str('%.2f' % TSIAout[0])+'  '+str('%.3f' % TSIAout[3]),
           'MIDAS: '+str('%.2f' % MIDASout[0])+'  '+str('%.3f' % MIDASout[3]),
           'Truth:   '+str('%.2f' % slope)])
leg.get_frame().set_alpha(1)
ax.grid('on')
plt.xlim(-150,time[-1]+50)
plt.ylim(-50,1.2*slope*time[-1]+bf)
XT=sp.arange(0,sp.ceil(time[-1]/365),1)
plt.xticks(XT*365,XT)
plt.xlabel('Time (years)')
plt.ylabel('Value')
plt.title(lab[k])
plt.savefig(lab[k]+'.png',dpi=400)
plt.close(fig)
