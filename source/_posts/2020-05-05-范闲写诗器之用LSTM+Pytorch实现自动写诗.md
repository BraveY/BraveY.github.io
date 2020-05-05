---
title: 范闲写诗器之用LSTM+Pytorch实现自动写诗
date: 2020-05-05 15:15:15
categories: 深度学习
tags:
- LSTM
- Pytorch
copyright: true
---

LSTM网络经常用于序列预测，因此在NLP领域很常用，本文将利用LSTM网络来搭建一个简单的自动写诗的demo，做这个的时候突然想起庆余年中范闲作诗的片段，所以就把它取名为范闲写诗器，用来供范闲参考哈哈哈。文章的源码放在[github](<https://github.com/BraveY/AI-with-code/tree/master/Automatic-poem-writing> )上，求个赞！！！

```python
import torch
import os
from torch import nn 
import numpy as np 
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
from tqdm.notebook import tqdm
# from tqdm import tqdm
```

## 相关配置

```python
class DictObj(object):
    # 私有变量是map
    # 设置变量的时候 初始化设置map
    def __init__(self, mp):
        self.map = mp
        # print(mp)

# set 可以省略 如果直接初始化设置
    def __setattr__(self, name, value):
        if name == 'map':# 初始化的设置 走默认的方法
            # print("init set attr", name ,"value:", value)
            object.__setattr__(self, name, value)
            return
        # print('set attr called ', name, value)
        self.map[name] = value
# 之所以自己新建一个类就是为了能够实现直接调用名字的功能。
    def __getattr__(self, name):
        # print('get attr called ', name)
        return  self.map[name]


Config = DictObj({
    'poem_path' : "./tang.npz",
    'tensorboard_path':'./tensorboard',
    'model_save_path':'./modelDict/poem.pth',
    'embedding_dim':100,
    'hidden_dim':1024,
    'lr':0.001,
    'LSTM_layers':3
})
```

## 唐诗数据查看

唐诗数据文件分为三部分，data部分是唐诗数据的总共包含57580首唐诗数据，其中每一首都被格式化成125个字符，唐诗开始用'<START\>'标志，结束用'<EOP\>'标志,空余的用'<space\>'标志， ix2word和word2ix是汉字的字典索引。因此可以不用自己去构建这个字典了。

```python
def view_data(poem_path):
    datas = np.load(poem_path)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    word_data = np.zeros((1,data.shape[1]),dtype=np.str) # 这样初始化后值会保留第一一个字符，所以输出中'<START>' 变成了'<'
    row = np.random.randint(data.shape[0])
    for col in range(data.shape[1]):
        word_data[0,col] = ix2word[data[row,col]]
    print(data.shape) #(57580, 125)
    print(word_data)#随机查看
```

```python
view_data(Config.poem_path)
```

```
(57580, 125)
[['<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<'
  '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<'
  '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<'
  '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<' '<'
  '<' '<' '<' '<' '庭' '树' '晓' '禽' '动' '，' '郡' '楼' '残' '点' '声' '。' '灯' '挑'
  '红' '烬' '落' '，' '酒' '煖' '白' '光' '生' '。' '髪' '少' '嫌' '梳' '利' '，' '颜' '衰'
  '恨' '镜' '明' '。' '独' '吟' '谁' '应' '和' '，' '须' '寄' '洛' '阳' '城' '。' '<']]
```

可以看到125个字符中，大部分都是空格数据，我统计了下总的数据中将近57%的数据都是空格。如果不去除空格数据的话，模型的虽然最开始训练的时候就有60多的准确率，但是这些准确率是因为预测空格来造成的，所以需要将空格数据给去掉。

## 构造数据集

这一步的主要工作是将原始的唐诗数据集中的空格给过滤掉，然后根据序列的长度seq_len重新划分无空格的数据集,并得到每个序列的标签。我最开始的时候比较困惑数据的标签应该是什么？纠结的点在于最开始以为上一句诗的标签是不是，下一句诗。比如“床前明月光，”的标签是不是“疑是地上霜”。后面阅读了些资料才搞懂正确的标签应该是这个汉字的下一个汉字，“床前明月光，”对应的标签是“前明月光，疑”。也就是基于字符级的语言模型。

[Dive-into-DL-Pytorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch/tree/master/docs/chapter06_RNN) 中的例子：

![](C:\Users\BraveY\Documents\BraveY\blog\images\深度学习\RNN.jpg)
搞懂了标签是什么后，这一步的代码逻辑就好理解了，先从路径文件中得到原始数据poem_data，然后将poem_data中的空格数据给过滤掉并平整到一维得到no_space_data，之后就根据索引去得到对应迭代次数的数据和标签了。



```python
class PoemDataSet(Dataset):
    def __init__(self,poem_path,seq_len):
        self.seq_len = seq_len
        self.poem_path = poem_path
        self.poem_data, self.ix2word, self.word2ix = self.get_raw_data()
        self.no_space_data = self.filter_space()
        
    def __getitem__(self, idx:int):
        txt = self.no_space_data[idx*self.seq_len : (idx+1)*self.seq_len]
        label = self.no_space_data[idx*self.seq_len + 1 : (idx+1)*self.seq_len + 1] # 将窗口向后移动一个字符就是标签
        txt = torch.from_numpy(np.array(txt)).long()
        label = torch.from_numpy(np.array(label)).long()
        return txt,label
    
    def __len__(self):
        return int(len(self.no_space_data) / self.seq_len)
    
    def filter_space(self): # 将空格的数据给过滤掉，并将原始数据平整到一维
        t_data = torch.from_numpy(self.poem_data).view(-1)
        flat_data = t_data.numpy()
        no_space_data = []
        for i in flat_data:
            if (i != 8292 ):
                no_space_data.append(i)
        return no_space_data
    def get_raw_data(self):
#         datas = np.load(self.poem_path,allow_pickle=True)  #numpy 1.16.2  以上引入了allow_pickle
        datas = np.load(self.poem_path)
        data = datas['data']
        ix2word = datas['ix2word'].item()
        word2ix = datas['word2ix'].item()
        return data, ix2word, word2ix
```

seq_len 这里我选择的是48，因为考虑到唐诗主要是五言绝句和七言绝句，各自加上一个标点符号也就是6和8，选择一个公约数48，这样刚好凑够8句无言或者6句七言，比较符合唐诗的偶数句对。

```python
poem_ds = PoemDataSet(Config.poem_path, 48)
ix2word = poem_ds.ix2word
word2ix = poem_ds.word2ix
```

```python
poem_ds[0]
```



```
(tensor([8291, 6731, 4770, 1787, 8118, 7577, 7066, 4817,  648, 7121, 1542, 6483,
         7435, 7686, 2889, 1671, 5862, 1949, 7066, 2596, 4785, 3629, 1379, 2703,
         7435, 6064, 6041, 4666, 4038, 4881, 7066, 4747, 1534,   70, 3788, 3823,
         7435, 4907, 5567,  201, 2834, 1519, 7066,  782,  782, 2063, 2031,  846]),
 tensor([6731, 4770, 1787, 8118, 7577, 7066, 4817,  648, 7121, 1542, 6483, 7435,
         7686, 2889, 1671, 5862, 1949, 7066, 2596, 4785, 3629, 1379, 2703, 7435,
         6064, 6041, 4666, 4038, 4881, 7066, 4747, 1534,   70, 3788, 3823, 7435,
         4907, 5567,  201, 2834, 1519, 7066,  782,  782, 2063, 2031,  846, 7435]))
```



```python
poem_loader =  DataLoader(poem_ds,
                     batch_size=16,
                     shuffle=True,
                     num_workers=0)
```

## 模型构造

模型使用embedding+LSTM来进行构造，embedding的理解参考 ，[Word2Vec](https://www.zybuluo.com/Dounm/note/591752)和[王喆的文章](https://zhuanlan.zhihu.com/p/53194407)。使用embedding层后将汉字转化为embedding向量，与简单使用One-hot编码可以更好地表示汉字的语义，同时减少特征维度。

向量化后使用LSTM网络来进行训练，LSTM的参数理解可能比较费劲，参考两张图进行理解：
[LSTM神经网络输入输出究竟是怎样的？ - Scofield的回答 - 知乎](https://www.zhihu.com/question/41949741/answer/318771336)中的
![](https://pic2.zhimg.com/80/v2-b45f69904d546edde41d9539e4c5548c_720w.jpg)
这张图中就能比较好理解input_size=embedding_dim,hidden_dim这两个个参数了。输入的X的维度就是图中绿色节点的数目了，在这里是embedding_dim这个参数就是经过向量化后的每个汉字，hidden_dim就是图中黄色的节点个数了，只不过因为LSTM有h和c两个隐藏状态，所以hidden_dim同时设置了h和c两个隐藏层状态的维度，也就是图中的黄色节点需要乘以2，变成两个的。num_layers的理解参考[LSTM细节分析理解（pytorch版） - ymmy的文章 - 知乎](https://zhuanlan.zhihu.com/p/79064602)的图片就是纵向的深度。 
![](https://pic4.zhimg.com/80/v2-ebf8cd2faa564d9d80a958dcf25e6b3b_720w.jpg)
LSTM的深度选择了3层，需要注意的是并不是越深越好，自己的初步实验显示加深到8层的效果并不比3层的效果好，因此选择了3层。

经过LSTM输出的维度也是hidden_dim，使用3层全连接来进一步处理。这里全连接层的激活函数选用的是tanh激活函数，因为初步的对比实验显示，tanh的效果比relu好一些。

```python
# import torch.nn.functional as F
class MyPoetryModel_tanh(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(MyPoetryModel_tanh, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)#vocab_size:就是ix2word这个字典的长度。
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                            batch_first=True,dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(self.hidden_dim,2048)
        self.fc2 = nn.Linear(2048,4096)
        self.fc3 = nn.Linear(4096,vocab_size)
#         self.linear = nn.Linear(self.hidden_dim, vocab_size)# 输出的大小是词表的维度，

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)  # [batch, seq_len] => [batch, seq_len, embed_dim]
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(Config.LSTM_layers*1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))#hidden 是h,和c 这两个隐状态
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = self.fc3(output)
        output = output.reshape(batch_size * seq_len, -1)
        return output,hidden
```

## 训练日志的设置

注释参考我之前的文章[Kaggle猫狗识别Pytorch详细搭建过程 - BraveY的文章 - 知乎](https://zhuanlan.zhihu.com/p/136421422)

```python
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
```

```python
## topk的准确率计算
def accuracy(output, label, topk=(1,)):
    maxk = max(topk) 
    batch_size = label.size(0)
    
    # 获取前K的索引
    _, pred = output.topk(maxk, 1, True, True) #使用topk来获得前k个的索引
    pred = pred.t() # 进行转置
    # eq按照对应元素进行比较 view(1,-1) 自动转换到行为1,的形状， expand_as(pred) 扩展到pred的shape
    # expand_as 执行按行复制来扩展，要保证列相等
    correct = pred.eq(label.view(1, -1).expand_as(pred)) # 与正确标签序列形成的矩阵相比，生成True/False矩阵
#     print(correct)

    rtn = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0) # 前k行的数据 然后平整到1维度，来计算true的总个数
        rtn.append(correct_k.mul_(100.0 / batch_size)) # mul_() ternsor 的乘法  正确的数目/总的数目 乘以100 变成百分比
    return rtn
```

## 迭代训练函数

训练中使用tensorboard来绘制曲线，终端输入`tensorboard --logdir=/path_to_log_dir/ --port 6006` 可查看

```python
def train( epochs, train_loader, device, model, criterion, optimizer,scheduler,tensorboard_path):
    model.train()
    top1 = AvgrageMeter()
    model = model.to(device)
    for epoch in range(epochs):
        train_loss = 0.0
        train_loader = tqdm(train_loader)
        train_loader.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
        for i, data in enumerate(train_loader, 0):  # 0是下标起始位置默认为0
            inputs, labels = data[0].to(device), data[1].to(device)
#             print(' '.join(ix2word[inputs.view(-1)[k] for k in inputs.view(-1).shape.item()]))
            labels = labels.view(-1) # 因为outputs经过平整，所以labels也要平整来对齐
            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()
            outputs,hidden = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            _,pred = outputs.topk(1)
#             print(get_word(pred))
#             print(get_word(labels))
            prec1, prec2= accuracy(outputs, labels, topk=(1,2))
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
            train_loader.set_postfix(log=postfix)
            
#             break
            # ternsorboard 曲线绘制
            if os.path.exists(Config.tensorboard_path) == False: 
                os.mkdir(Config.tensorboard_path)    
            writer = SummaryWriter(tensorboard_path)
            writer.add_scalar('Train/Loss', loss.item(), epoch)
            writer.add_scalar('Train/Accuracy', top1.avg, epoch)
            writer.flush()
        scheduler.step()

    print('Finished Training')
```

## 模型初始化

初始化模型，并选择损失函数和优化函数。需要注意的是自己在训练过程中发现会发生loss上升的情况，这是因为到后面lr学习率过大导致的，解决的办法是是使用学习率动态调整。这里选择了步长调整的方法，每过10个epoch，学习率调整为原来的0.1。pytorch还有许多其他调整方法，参考这篇[文章](https://www.jianshu.com/p/26a7dbc15246)

初始学习率设置为0.001，稍微大点变成0.01最开始的训练效果都比较差，loss直接上百，不下降。

```python
# 上述参数的配置网络训练显存消耗为2395M，超过显存的话，重新调整下网络配置
model = MyPoetryModel_tanh(len(word2ix),
                  embedding_dim=Config.embedding_dim,
                  hidden_dim=Config.hidden_dim)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 30
optimizer = optim.Adam(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10,gamma=0.1)#学习率调整
criterion = nn.CrossEntropyLoss()
```

```python
model
```



```
MyPoetryModel_tanh(
  (embeddings): Embedding(8293, 100)
  (lstm): LSTM(100, 1024, num_layers=3, batch_first=True)
  (fc1): Linear(in_features=1024, out_features=2048, bias=True)
  (fc2): Linear(in_features=2048, out_features=4096, bias=True)
  (fc3): Linear(in_features=4096, out_features=8293, bias=True)
)
```



# 训练

```python
# model.load_state_dict(torch.load(Config.model_save_path))  # 模型加载
```



```
<All keys matched successfully>
```



```python
#因为使用tensorboard画图会产生很多日志文件，这里进行清空操作
import shutil  
if os.path.exists(Config.tensorboard_path):
    shutil.rmtree(Config.tensorboard_path)  
    os.mkdir(Config.tensorboard_path)
```

```python
train(epochs, poem_loader, device, model, criterion, optimizer,scheduler, Config.tensorboard_path)
```

```python
#模型保存
if os.path.exists(Config.model_save_path) == False: 
    os.mkdir(Config.model_save_path)   
torch.save(model.state_dict(), Config.model_save_path)
```

## 模型使用

使用训练好的模型来进行自动写诗创作，模型训练了30多个epoch，'train_loss': '0.452125', 'train_acc': '91.745990'

```python
model.load_state_dict(torch.load(Config.model_save_path))  # 模型加载
```



```
<All keys matched successfully>
```



生成的逻辑是输入一个汉字之后，给出对应的预测输出，如果这个输出所在的范围在给定的句子中，就摒弃这个输出，并用给定句子的下一个字做为输入，
直到输出的汉字超过给定的句子范围，用预测的输出句子作为下一个输入。
因为每次模型输出的还包括h和c两个隐藏状态，所以前面的输入都会更新隐藏状态，来影响当前的输出。也就是hidden这个tensor是一直在模型中传递，只要没有结束

```python
def generate(model, start_words, ix2word, word2ix,device):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    
    #最开始的隐状态初始为0矩阵
    hidden = torch.zeros((2, Config.LSTM_layers*1,1,Config.hidden_dim),dtype=torch.float)
    input = input.to(device)
    hidden = hidden.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
            for i in range(48):#诗的长度
                output, hidden = model(input, hidden)
                # 如果在给定的句首中，input为句首中的下一个字
                if i < start_words_len:
                    w = results[i]
                    input = input.data.new([word2ix[w]]).view(1, 1)
               # 否则将output作为下一个input进行
                else:
                    top_index = output.data[0].topk(1)[1][0].item()#输出的预测的字
                    w = ix2word[top_index]
                    results.append(w)
                    input = input.data.new([top_index]).view(1, 1)
                if w == '<EOP>': # 输出了结束标志就退出
                    del results[-1]
                    break
    return results
```

```python
results = generate(model,'雨', ix2word,word2ix,device)
print(' '.join(i for i in results))
```

```
雨 余 芳 草 净 沙 尘 ， 水 绿 滩 平 一 带 春 。 唯 有 啼 鹃 似 留 客 ， 桃 花 深 处 更 无 人 。
```



```python
results = generate(model,'湖光秋月两相得', ix2word,word2ix,device)
print(' '.join(i for i in results))
```

```
湖 光 秋 月 两 相 得 ， 楚 调 抖 纹 难 自 干 。 唱 至 公 来 尊 意 敬 ， 为 君 急 唱 曲 江 清 。
```



```python
results = generate(model,'人生得意须尽欢，', ix2word,word2ix,device)
print(' '.join(i for i in results))
```

```
人 生 得 意 须 尽 欢 ， 吾 见 古 人 未 能 休 。 空 令 月 镜 终 坐 我 ， 梦 去 十 年 前 几 回 。 谁 谓 一 朝 天 不 极 ， 重 阳 堪 发 白 髭 肥 。

```



```python
results = generate(model,'万里悲秋常作客，', ix2word,word2ix,device)
print(' '.join(i for i in results))
```

```
万 里 悲 秋 常 作 客 ， 伤 人 他 日 识 文 诚 。 经 时 偏 忆 诸 公 处 ， 一 叶 黄 花 未 有 情 。

```



```python
results = generate(model,'风急天高猿啸哀，渚清沙白鸟飞回。', ix2word,word2ix,device)
print(' '.join(i for i in results))
```

```
风 急 天 高 猿 啸 哀 ， 渚 清 沙 白 鸟 飞 回 。 孤 吟 一 片 秋 云 起 ， 漏 起 傍 天 白 雨 来 。

```



```python
results = generate(model,'千山鸟飞绝，万径人踪灭。', ix2word,word2ix,device)
print(' '.join(i for i in results))
```

```
千 山 鸟 飞 绝 ， 万 径 人 踪 灭 。 日 暮 沙 外 亭 ， 自 思 林 下 客 。

```

## 参考

[Dive-into-DL-Pytorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch/tree/master/docs/chapter06_RNN) RNN章节