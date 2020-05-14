import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from linear_data import generate_linear_data
from itertools import combinations 
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StandardizedLinearRegression:
    def __init__(self, X, y, beta):
        self.X=X
        self.y=y
        self.beta=beta
        self.C = self.X.T.dot(self.X)
        self.r = self.X.T.dot(self.y)
        self.beta_estimate = np.linalg.inv(self.C).dot(self.r)
    def set_XY(self, X, y):
        self.X=X
        self.y=y
        self.C = self.X.T.dot(self.X)
        self.r = self.X.T.dot(self.y)
        if(len(self.C.shape)):
            self.beta_estimate = np.linalg.inv(self.C).dot(self.r)
        else:
            self.beta_estimate = (1/self.C)*self.r
    def solve_linear_regression(self, X=None, y=None):
        if(X!=None and y!=None):
            self.X=X
            self.y=y
            self.beta=beta
            self.C = self.X.T.dot(self.X)
            self.r = self.X.T.dot(self.y)

        self.beta_estimate = np.linalg.inv(self.C).dot(self.r)
        return self.beta_estimate
    
    def RSS(self):
        '''
        Residual sum of squares
        '''
        error=self.y - self.X.dot(self.beta_estimate)
        return (error).T.dot(error)
    
    def TSS(self):
        '''
        RSS for an intercept model
        $(Y = \beta_0)$
        '''
        return (y-y.mean()).T.dot(y-y.mean())
    def r2(self):
        return 1 - self.RSS()/self.TSS()
    
    def net_effect(self):
        return self.beta_estimate*self.r

def calculate_R2_j(X,y,SLR):
    '''
    Calculate $R^2_{j,-j}$
    R^2 for the model
    x_j = \beta_1x_1 + .. \beta_{j-1}x_{j-1} + \beta_{j+1}x_{j+1} + ... + \beta_{n}x_{n}
    
    Predict x_j with all other n-1 predictors
    '''
    n=X.shape[1]
    r2_n_self=[]
    for i in range(n):
        X_new = np.concatenate((X[:, :i], X[:,i+1:]), axis=1)
        y_new = X[:, i]
        SLR.set_XY(X_new, y_new)
        r2_n_self.append(SLR.r2())
    r2_n_self=np.array(r2_n_self)
    return r2_n_self

def calculate_Ry_j(X,y,SLR):
    '''
    Calculate R^2_{y,-j}
    R^2 for the model
    y = \beta_1x_1 + .. \beta_{j-1}x_{j-1} + \beta_{j+1}x_{j+1} + ... + \beta_{n}x_{n}

    All predictors except predictor j
    '''
    n=X.shape[1]
    r2_y_self=[]
    for i in range(n):
        X_new = np.concatenate((X[:, :i], X[:,i+1:]), axis=1)
        y_new = y
        SLR.set_XY(X_new, y_new)
        r2_y_self.append(SLR.r2())
    r2_y_self=np.array(r2_y_self)

def calculate_R2_i_j(X,y,X_idx,SLR):
    '''
    X_idx=[i1,i2,...,ik] (list of index of predictors to use)
    Then calculate R^2 for the model
    y = \beta_{i1}x_{i1} + \beta_{i2}x_{i2} + .. +\beta_{ik}x_{ik}
    '''
    X_idx=X_idx.astype('int')
    X=X[:, X_idx]
    SLR.set_XY(X,y)
    return SLR.r2()

def calculate_SV_j(X,y,selectIdx,SLR):
    '''
    Calculate the Incremental Net Effect using Shapley Values
    for a given predictor selectIdx
    '''
    n=X.shape[1]
    idx=np.arange(n)
    del_idx=np.delete(idx,selectIdx)
    SLR.set_XY(X,y)
    sum_coalation=SLR.r2()/n
    for i in range(1,n):
        contribution_i=[]
        overall_contribution=[]
        if(i==1):
            first_comb=np.array([selectIdx])
        else:
            first_comb = np.array(list  (combinations(del_idx, i-1)))
            first_comb=np.c_[np.ones(first_comb.shape[0])*selectIdx, first_comb]
        second_comb= np.array(list(combinations(idx, i)))
        for first in first_comb:
            r=calculate_R2_i_j(X,y,first,SLR)
            contribution_i.append(r)
        contribution_i=np.array(contribution_i)
        for second in second_comb:
            r=calculate_R2_i_j(X,y,second,SLR)
            overall_contribution.append(r)
        overall_contribution=np.array(overall_contribution)

        mean_diff = contribution_i.mean() - overall_contribution.mean()
        sum_coalation+=mean_diff/(n-i)
    return sum_coalation

def nCr(n,r):
    '''
    nCr = n!/(r! (n-r)!)
    '''
    f = math.factorial
    return f(n) // f(r) // f(n-r)
N=1500
n=10
X,y,beta=generate_linear_data(N,n,n_rel=-1, data_rel_var=2.0 ,data_irrel_var=5.0,\
                                        noise_var=5.0,standardized=True,correlated=True)

SLR = StandardizedLinearRegression(X,y,beta)
beta_estim = SLR.solve_linear_regression()
r2_prob = SLR.r2()
net_effects=SLR.net_effect()


fig = make_subplots(rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing = 0.05,
        subplot_titles=("Net Effects", "Predictor Estimated Coefficients"))

fig.add_trace(go.Scatter(x=np.arange(n), y=net_effects, mode='lines+markers',
                        name='NEF'), row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(n), y=beta_estim,
                    mode='lines+markers',
                    name='LS Coeff'), row=2, col=1)
fig.update_layout(
    title={
        'text': "R2 = " + str(r2_prob),
    },
    height=950,
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#000000"
    )
)
for i in fig['layout']['annotations']:
    i['font'] = dict(size=23,color='#000000')

fig.show()
# fig.write_html("/home/tlokeshkumar/Documents/IITM/devlopr-jekyll/_includes/_plots/neteffects/netEffectsCoeff.html", include_plotlyjs="cdn")


print("Max Combinations ", nCr(n, n//2))
SV=[]
for i in range(n):
    SV.append(calculate_SV_j(X,y,i,SLR))
# plt.plot(SV, label='INEF')
# plt.plot(net_effects, label='NEF')
# plt.legend(loc='best')
# plt.show()
print(SV)
print(net_effects)

print('r2 ', r2_prob)
print('r2_sv ', np.sum(SV) )

fig = make_subplots(rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing = 0.05,
        subplot_titles=("Net Effects & Incremental Net Effects (Shapely Values)", "Predictor Estimated Coefficients"))

fig.add_trace(go.Scatter(x=np.arange(n), y=net_effects, mode='lines+markers',
                        name='NEF'), row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(n), y=SV,
                    mode='lines+markers',
                    name='Shapley Values'), row=1, col=1)

fig.add_trace(go.Scatter(x=np.arange(n), y=beta_estim,
                    mode='lines+markers',
                    name='LS Coeff'), row=2, col=1)
fig.update_layout(
    title={
        'text': "R2 = " + str(r2_prob),
    },
    height=950,
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#000000"
    )
)
for i in fig['layout']['annotations']:
    i['font'] = dict(size=23,color='#000000')

fig.show()
fig.write_html("/home/tlokeshkumar/Documents/IITM/devlopr-jekyll/_includes/_plots/neteffects/netEffectsSV.html", include_plotlyjs="cdn")
# fig.write_html("netEffectsCoeffCorrel.html", include_plotlyjs="cdn")

# inc_r_2 = beta_estim**2 *(1-r2_n_self)  #U_j
# # print(inc_r_2+r2_y_self)
# R2_inc = np.mean(inc_r_2+r2_y_self)
# print("R2_inc ", R2_inc)
# print('r2 ', r2_prob)
# print('r2 sklearn ', r2_score(y, X.dot(beta_estim)))

   