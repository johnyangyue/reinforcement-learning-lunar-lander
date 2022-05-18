import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as T

def plot_scores(x_axis,scores,avg_scores,min_scores,max_scores,epsilon,filename):
    n= len(scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = np.mean(scores[max(0,t-100):(t+1)])

    f,a = plt.subplots()
    sns.lineplot(x=x_axis,y=scores,label='Raw',ax=a,color='deepskyblue')
    sns.lineplot(x=x_axis,y=avg_scores,label='Average',ax=a,linestyle='dashdot',color='darkorange')
    #sns.lineplot(x=x_axis,y=min_scores,label='Minimum',ax=a,linestyle='dashed',color='dimgray')
    #sns.lineplot(x=x_axis,y=max_scores,label='Maximum',ax=a,linestyle='dashdot',color='dimgrey')
    a.set_xlabel('Training Steps/Episodes')
    a.set_ylabel('Scores')
    plt.legend(bbox_to_anchor=(1.1,1),loc='upper left')

    a_alt = a.twinx()
    sns.lineplot(x=x_axis,y=epsilon,color='aqua',linestyle='dotted')
    a_alt.set_ylabel('Epsilons')

    plt.title('Training History')
    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight',dpi=200)

def get_screen_basic(render,shape):
    screen = render
    screen = np.ascontiguousarray(screen,dtype=np.float32)
    screen = cv2.cvtColor(screen,cv2.COLOR_RGB2GRAY)
    screen = cv2.resize(screen,shape[1:],interpolation=cv2.INTER_AREA)
    screen = np.array(screen,dtype=np.bool).reshape(shape)
    return screen/255


