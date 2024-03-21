import datetime
import os

from BaseDetectors.basedetector_TNR import base_detector

if __name__ == '__main__':
    start = datetime.datetime.now()

    anomalies = [0]
    model_name = ['ComplEx']  # "TransE", "ComplEx", "DistMult"

    kk = [50]
    for s in range(len(kk)):
        for i in range(len(anomalies)):
            for j in range(len(model_name)):
                scoredict, triplelist = base_detector(model_name[j], anomalies[i],
                                                      kk[s])

    end = datetime.datetime.now()

    print(start)
    print(end)

    modelname = 'ComplEx'
    dataset = 'UMLS'  # UMLS WN FB

    outpath = './ranking/' + dataset

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # print(len(scoredict))
    # print(len(triplelist))
    outpath = outpath + '/' + modelname + '.txt'
    with open(outpath, 'w') as f:
        print('--------Ranking all triples-------')
        ranking = sorted(scoredict.items(), key=lambda x: x[1], reverse=True)
        print('--------Writing to file-------')
        a = 0
        for i in range(len(ranking)):
            # if a <= 16281:
            # print('Correct')
            if ranking[i][0] in triplelist.keys():
                # f.write(str(ranking[i][0]).replace('[','').replace(']','').replace(' ','').replace("'",''))
                f.write(
                    str(triplelist[ranking[i][0]]).replace('[', '').replace(
                        ']', '').replace(' ', '').replace("'", ''))
                f.write(',')
                f.write(str(ranking[i][1]))
                f.write('\n')
                a += 1

    ##########################################################################################

    dictionary = {}
    rawpath = './raw/' + dataset + '-raw.txt'
    rankingpath = outpath
    writepath = './labeled/' + dataset
    if not os.path.exists(writepath):
        os.makedirs(writepath)
    writepath = writepath + '/' + modelname + '.txt'

    dict = open(rawpath)
    for line in dict.readlines():
        x, y, z, label = line.strip('\n').split(' ')
        a = [x, y, z]
        dictionary[str(a)] = label
        # print(dictionary)
    # print(dictionary)

    print('----------------Writing to the file-------------')
    with open(writepath, 'w') as f:
        aaa = open(rankingpath)
        for line in aaa.readlines():
            x, y, z, b = line.strip('\n').split(',')
            if str([x, y, z]) in dictionary.keys():
                # print(str([x, y, z]))
                # print(dictionary[str([x, y, z])])
                a = str([x, y, z]).replace('[', '').replace(']', '').replace(
                    ' ', '').replace('"', '').replace("'", '')
                f.write(a)
                f.write(',')
                f.write(dictionary[str([x, y, z])])
                f.write('\n')
