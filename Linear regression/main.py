import numpy as np
lr=0.0000000000001
train_x=[]
train_y=[]
with open("week1train.txt",'r') as a:
    for line in a:
        train_x.append(float(line.split()[0]))
        train_y.append(float(line.split()[1]))
train_x=np.array(train_x)
train_y=np.array(train_y)
a,b=4.234567800005467,8.765941999628708
best=1
best_a,best_b=0,0
c=0
while True:
    c+=1
    g_a=0.0
    g_b=0.0
    g=0.0
    for i,j in zip(train_x,train_y):
        g+=(j-a*i-b)**2
    if g<best:
        best=g
        best_a=a
        best_b=b
        print(best,best_a,best_b)
    for i,j in zip(train_x,train_y):
        g_a=g_a+(j-a*i-b)*(-i)
        g_b=g_b+(j-a*i-b)*(-1)
    a=a-g_a*lr
    b=b-g_b*lr


