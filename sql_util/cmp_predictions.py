import pickle as pkl

better = pkl.load(open('better.pkl', 'rb'))
worse = pkl.load(open('worse.pkl', 'rb'))

for i, (b, w) in enumerate(zip(better, worse)):
    if int(b) != int(w):
        print(b, w)
        print(i)


