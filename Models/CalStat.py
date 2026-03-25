from DataLoaderSkin import *

dataset = 'ISIC2018'
itrNo = 1
fnArr, lblArr = getData(dataset, 'train', itrNo)
fnArr_v, lblArr_v = getData(dataset, 'val', itrNo)
fnArr_test, lblArr_test = getData(dataset, 'test', itrNo)
fnArr = fnArr + fnArr_v + fnArr_test
lblArr = lblArr + lblArr_v + lblArr_test
fnArr = np.array(fnArr)
lblArr = np.array(lblArr)

dataset = DatasetSkin(False, fnArr, lblArr)
bs = 64
loader = DataLoader(
    dataset,
    shuffle=True, #sampler=SequentialSampler(data_T), #
    batch_size=bs,
    num_workers=8)


mean = 0.
std = 0.
nb_samples = 0.
for i, (data,_,_,_) in enumerate(loader):
    if data.shape[0]!=bs:
        break
    if i%100==0:
        print(i)
    data = data.cuda().float()
    batch_samples = data.size(0)
    if data.shape[1] == 1:
        mean += data.mean()
        std += data.std()
    else:
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)



