import pickle5 as pickle
import gzip, numpy
from multiprocessing import Pool
from itertools import product
from PIL import Image


number_of_pixels = 784


def activation(input):
    if(input>0):
        return 1
    return 0

# INITIALISATION
def init():
    neurons=[]
    trained=0
    if trained==0:
        for i in range(10):
            w=[0]*number_of_pixels
            t=[0]*10
            t[i]=1
            x=[w,t,0]
            neurons.append(x)
    return neurons

# TRAINING
def train(neurons,iterations,i,train_set):
    ok=False
    while(ok==False and iterations>0):
        iterations-=1
        ok=True
        n=len(train_set[0])
        test1 = 0
        for j in range(n):
            if j%500==0 and i==0:
                print("training iteration left =",iterations+1,"         ",j/500,"% completed")
            z = neurons[2]
            for k in range(number_of_pixels):
                z+=neurons[0][k]*train_set[0][j][k]
            output = activation(z)
            for k in range(number_of_pixels):
                neurons[0][k]+=(neurons[1][train_set[1][j]]-output)*train_set[0][j][k]*0.01
                neurons[2]+=(neurons[1][train_set[1][j]]-output)*0.01
            if output != neurons[1][train_set[1][j]]:
                test1+=1
                ok=False
        print(i,test1)
    return neurons
    # //////////  testing
def test(neurons,test_set):
    print("Testing...")
    n=len(test_set[0])
    error = 0
    for j in range(n):
        if j%100==0:
            print(j/100,"%")
        results = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(10):
            z = neurons[i][2]
            for k in range(number_of_pixels):
                z+=neurons[i][0][k]*test_set[0][j][k]
            results[i]=z
        x=results.index(max(results))
        if x != test_set[1][j]:
            error += 1
    print("Recognises the digit ",(n-error)/(n/100),"% of the time")

if __name__ == '__main__':
    f = gzip.open('C:\\Users\Dan\PycharmProjects\\NeuralNetworks2\mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin')
    f.close()
    iterations=10
    initializat=1
    save=0
    viz=1
    training=0
    if initializat==0:
        neurons=init()
    else:
        f = open('store.pckl', 'rb')
        neurons = pickle.load(f)
        f.close()
    print(neurons)
    arguments=[]
    for i in range(10):
        a=neurons[i]
        x=[a,iterations,i,train_set]
        arguments.append(x)
    if training==1:
        with Pool(10) as p:
            neurons=p.starmap(train, arguments)
    if viz==1:
        for i in range(10):
            a = numpy.array(neurons[i][0])
            m=max(a)
            for j in range(number_of_pixels):
                a[j] *=(255/m)
            a=a+64
            a = a.reshape(28, 28)
            img = Image.fromarray(a)
            img = img.resize((1000, 1000), Image.ANTIALIAS)
            img = img.convert('RGB')
            img.save("C:\\Users\Dan\Desktop\\neurons\\" + str(i) + ".png")
    test(neurons,test_set)
    if save==1:
        f = open('store.pckl', 'wb')
        pickle.dump(neurons, f)
        f.close()