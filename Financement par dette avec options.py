
import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

data = pd.read_csv('data_TP2_H2020.csv')  # Veuillez prendre note que nous avons ajoute une ligne ('Price') dans le fichier csv pour pas que notre array saute le premier prix


######### Question 1 ############



cap_struct = pd.DataFrame(np.array([[50, 300, 1500], [60, 330, 1600]]).T,
                          columns=['Open', 'Close'],
                          index=["Nmb d'actions en circulations", 'Dettes CT', 'Dettes LT'])

# Methode KMV

r = 0.02
T = 1

coeff_equity = (cap_struct.iloc[0, 1] - cap_struct.iloc[0, 0]) / (len(data) - 1)
coeff_debt_CT = (cap_struct.iloc[1, 1] - cap_struct.iloc[1, 0]) / (len(data) - 1)
coeff_debt_LT = (cap_struct.iloc[2, 1] - cap_struct.iloc[2, 0]) / (len(data) - 1)

D = []
S = []

for i in range(0, len(data)):
    D = np.append(D, cap_struct.iloc[1, 0] + (coeff_debt_CT * i) + 0.5 * (cap_struct.iloc[2, 0] + (coeff_debt_LT * i)))
    S = np.append(S, data.iloc[i, 0] * (cap_struct.iloc[0, 0] + coeff_equity * i))

return_ = []
for i in range(1, len(data)):
    return_ = np.append(return_, np.log((S[i]) / (S[i-1])))

sigma = np.std(return_) * np.sqrt(len(data))
print('La volatilite des log rendements est : {:.2f} %'.format(sigma * 100))

data = data.iloc[1 :]
data['Equity'] = S[1 :]
data['Return'] = return_

A = D + S

M = cap_struct.iloc[1,1] + 0.5 * cap_struct.iloc[2,1]

# Graphique

plt.subplot(211)
plt.plot(data.iloc[:, 0])
plt.title('Cours boursiers du stockprice - 2020')
plt.xlabel('Days')
plt.ylabel('Price')
plt.subplot(212)
plt.plot(data.iloc[:, 2])
plt.title('Rendements du stock - 2020')
plt.xlabel('Days')
plt.ylabel('Return')
plt.savefig('Cours boursiers du stockprice - 2020')
plt.show()


# Calcul de la difference entre le prix de l'action theorique vs le prix observe

def Merton(A, M, S, r, T, sigma):
    d1 = (np.log(A / M) + (r + 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(A / M) + (r - 1 / 2 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    N_d1 = sp.norm.cdf(d1)
    N_d2 = sp.norm.cdf(d2)
    S_theorical = A * N_d1 - D * N_d2 * np.exp(-r * T)
    diff = S - S_theorical
    MertonDelta_ = -N_d1

    return MertonDelta_, diff


# Calcul de la valeur implicite des actifs

def A_implicite(A, D, S, r, T, sigma):
    epsilon = 0.0001
    nmax = 50  # nombre max d'iteration

    for m in range(0, nmax):
        MertonDelta_, diff = Merton(A, D, S, r, T, sigma)
        new_A = A - (diff / MertonDelta_)
        if np.abs(A - new_A) < epsilon:
            m = nmax
        else:
            A = new_A

    A_implicite = A

    return A_implicite


# calcul de la valeur implicite des actifs

max_ite = 10
diff_sigma_conf = 0.0001
sigma_KMV = np.array([sigma])


for i in range(0, max_ite):
    A_ = []
    for j in range(0, len(data)) :
        A_ = np.append(A_, A_implicite(A[j], D[j], S[j], r, T, sigma_KMV[i]))
    return_KMV = []
    for x in range(0, len(data) - 1):
        return_KMV= np.append(return_KMV, np.log(A_[x] / A_[x + 1]))
    sigma_KMV = np.append(sigma_KMV, np.sqrt(np.var(return_KMV)) * np.sqrt(len(data)))
    A = A_
    if np.abs(sigma_KMV[i+1] - sigma_KMV[i]) <= diff_sigma_conf :
        asset_vol = sigma_KMV[i+1]
        break
    else :
        continue


############## Question 2 ###############

## DATA ##

q = 0.3

r = 0.02
T = 10
n = T * 252
h = T/n
sigma = sigma_KMV[-1]
theta = 1
y = 0

v = cap_struct.iloc[0, 1] * data.iloc[-1, 0] + cap_struct.iloc[1, 1] + cap_struct.iloc[2, 1]
M = cap_struct.iloc[2, 1]


# Calcul de la valeur des actifs avec arbre binomial

def binomial_three(v, sigma, n, h):
    u = np.exp(sigma * np.sqrt(h))
    d = np.exp(-sigma * np.sqrt(h))
    s = (n * 2 + 1, n + 1)
    arbre = np.zeros(s)
    z = int(s[0] / 2 - 0.5)
    arbre[z, 0] = v
    for i in range(1, n + 1):
        arbre[z - i, i] = v * u ** i
        arbre[z + i, i] = v * d ** i
        for j in range(i + 1, n + 1):
            arbre[z - i + j - i, j] = arbre[z - i, i] * d ** (j - i)
    return arbre


def option_CD(three,q,M,sigma,theta,r,y,n) :

    option_three = three.copy()

    for i in range (0, n*2 + 1) :
        if i % 2 == 0 :
            x = three[i, -1]
            if x >= M/q :
                option_three[i, -1] = x * q
            elif M/q > x >= M :
                option_three[i, -1] = M
            else :
                option_three[i, -1] = theta * x
        else :
            option_three[i, -1] = 0

    u = np.exp(sigma)
    d = np.exp(-sigma)
    pi = (np.exp((r-y)) - d) / (u-d)

    for i in range (1, n + 1) :
        for j in range (0, n-i+1) :
            option_three[i + j * 2, n - i] = np.maximum((pi * option_three[i-1+j*2,n-i+1] + (1-pi) * option_three[i+1+j*2,n-i+1]) * np.exp(-r), three[i+j*2, n-i] * q)

    return option_three


three = binomial_three(v, sigma, n, h)
Dette_convertible = option_CD(three,q,M,sigma,theta,r,y,n)
print('La valeur de la dette convertible est : {:.2f} $'.format(Dette_convertible[n,0]))


## Structure par terme des taux de rendement

Dette_convertible_array = []
return_dette_conv = []

for t in (np.linspace(1,10,10)) :
    n = np.int(t) * 252
    h = t/n
    three = binomial_three(v, sigma, n, h)
    Dette_convertible_ = option_CD(three, q, M, sigma, theta, r, y, n)
    Dette_convertible_array = np.append(Dette_convertible_array, Dette_convertible_[n,0])
    return_dette_conv = np.append(return_dette_conv, np.log(M / Dette_convertible_array[np.int(t) - 1]) / np.sqrt(np.int(t)))\


############## Question 3 ###############



callable_price = 1550

##########  Fonction   ###############


def Merton_Debt(v, M, r, n, sigma, theta):
    d1 = (np.log(v / M) + (r + 1 / 2 * sigma ** 2) * n) / (sigma * np.sqrt(n))
    d2 = (np.log(v / M) + (r - 1 / 2 * sigma ** 2) * n) / (sigma * np.sqrt(n))
    N_d1 = sp.norm.cdf(d1)
    N_d2 = sp.norm.cdf(d2)
    S = v * np.exp(-y * n) * N_d1 - M * N_d2 * np.exp(-r * n)
    D = M * np.exp(-r * n) * N_d2 + theta * v * np.exp(-y * n) * (1 - N_d1)

    return D


def callable_option(three, callable_price, M, n, r, sigma, theta, h, y) :

    option_three = three.copy()

    for i in range(0, n+1):
        option_three[i*2, n] = np.minimum(Merton_Debt(three[i*2, n], M, r, h, sigma, theta), callable_price)

    u = np.exp(sigma)
    d = np.exp(-sigma)
    pi = (np.exp((r - y) * h) - d) / (u - d)

    for i in range (1, n+1) :
        for j in range (0, n-i+1) :
            option_three[i+j*2, n-i] = np.minimum((pi*option_three[i+j*2-1, n-i+1] + (1-pi)*option_three[i+j*2+1, n-i+1]) * np.exp(-r*h), callable_price)

    return option_three



### Calcul

three = binomial_three(v, sigma, n, h)
Dette_rachetable = callable_option(three, callable_price, M, n, r, sigma, theta, h, y)
print('La valeur de la dette rachetable est : {:.2f} $'.format(Dette_rachetable[n,0]))


## Structure par terme des taux de rendement

Dette_callable_array = []
return_dette_call  = []

for t in (np.linspace(1,10,10)) :
    n = np.int(t) * 252
    h = t / n
    three = binomial_three(v, sigma, n, h)
    Dette_callable_ = callable_option(three, callable_price, M, n, r, sigma, theta, h, y)
    Dette_callable_array = np.append(Dette_callable_array, Dette_callable_[n,0])
    return_dette_call  = np.append(return_dette_call , np.log(M/Dette_callable_array[np.int(t) - 1]) / np.sqrt(np.int(t)))




############## Question 4 ###############


# Dette convertible rachetable



def option_Conv_Call(three,q,M,sigma,theta,r,y,n) :

    option_three = three.copy()

    for i in range (0, n*2 + 1) :
        if i % 2 == 0 :
            x = three[i, -1]
            if x >= M/q :
                option_three[i, -1] = x * q
            elif M/q > x >= M :
                option_three[i, -1] = M
            else :
                option_three[i, -1] = theta * x
        else :
            option_three[i, -1] = 0

    u = np.exp(sigma)
    d = np.exp(-sigma)
    pi = (np.exp((r-y)*h) - d) / (u-d)

    for i in range (1, n + 1) :
        for j in range (0, n-i+1) :
            option_three[i + j * 2, n - i] = np.minimum(np.maximum((pi * option_three[i-1+j*2,n-i+1] + (1-pi) * option_three[i+1+j*2,n-i+1]) * np.exp(-r*h), three[i+j*2, n-i] * q), np.maximum(callable_price, three[i+j*2, n-i] * q))


    return option_three

Dette_CC = option_Conv_Call(three,q,M,sigma,theta,r,y,n)
print('La valeur de la dette convertible rachetable est : {:.2f} $'.format(Dette_CC[n,0]))


## Structure par terme des taux de rendement

Dette_CC_array = []
return_dette_conv_call = []

for t in (np.linspace(1,10,10)) :
    n = np.int(t) * 252
    h = t / n
    three = binomial_three(v, sigma, n, h)
    Dette_CC_ = option_Conv_Call(three,q,M,sigma,theta,r,y,n)
    Dette_CC_array = np.append(Dette_CC_array, Dette_CC_[n,0])
    return_dette_conv_call  = np.append(return_dette_conv_call , np.log(M/Dette_CC_array[np.int(t) - 1]) / np.sqrt(np.int(t)))





df = pd.DataFrame({'days': np.linspace(1, 10, 10), 'Dette convertible': return_dette_conv, 'Dette rachetable': return_dette_call,
                   'Dette convertible rachetable': return_dette_conv_call})

# multiple line plot


plt.plot('days', 'Dette convertible', data=df, color='black')
plt.plot('days', 'Dette rachetable', data=df, color='grey')
plt.plot('days', 'Dette convertible rachetable', data=df, color='red')
plt.title("Rendements des dettes selon leur option")
plt.legend()
plt.savefig("Rendements des dettes selon leur option")
plt.show()


plt.plot('days', 'Dette convertible', data=df, color='black')
plt.title('Dette convertible')
plt.xlabel('Echeance')
plt.ylabel('Return')
plt.savefig('Dette convertible')

plt.plot('days', 'Dette rachetable', data=df, color='black')
plt.title('Dette rachetable')
plt.xlabel('Echeance')
plt.ylabel('Return')
plt.savefig('Dette rachetable')

plt.plot('days', 'Dette convertible rachetable', data=df, color='black')
plt.title('Dette convertible rachetable')
plt.xlabel('Echeance')
plt.ylabel('Return')
plt.savefig('Dette convertible rachetable')





