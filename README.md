# Sentiment Analysis with Transformer

## Fork
https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch

## 수정사항
- [x] *.ipynb files -> *.py files  
- [x] numpy를 제외한 모든 작업을 gpu로 이관  
- [ ] tqdm_notebook -> tqdm  
- [x] checkpoint 추가
## Experiments
### Dataset
IMDb 영화 리뷰 데이터셋 50,000개 이용  
Train : Validation : Test = 0.7 : 0.2 : 0.1  
<img src="https://user-images.githubusercontent.com/77797199/123633685-75caa780-d854-11eb-877c-8181c05cf25d.PNG">  

### How to Use  
1. Install Requirements  
python version : 3.7 권장  
```
pip install requirements.txt
```
  
2. Download Dataset
```
# In Ubuntu
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb.tar.gz
```
  
압축 해제시 다음과 같은 구조의 폴더 생성  
```
aclImdb
  ├── test
  │     ├── pos
  │     ├── neg
  ├── train
        ├── pos
        └── neg
```

3. Run 
```
python data_processing.py
python transformer.py
```


### Result
<img src="https://user-images.githubusercontent.com/77797199/123633738-8d099500-d854-11eb-84dd-2fa9aef1ae8f.PNG">  
  
|    Model    | Train acc | Val acc | Test acc |
|:-----------:|:---------:|:-------:|:--------:|
| Transformer |   0.963   |  0.937  |  0.933   |   
  

## Main Idea
1. Transformer  
   > [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)  
   
   <img src = "https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/raw/013c55eaea3b4333a6b05a46ee1123811285a8f4/assets/Transformer-classifier.png">  
     
   > model/image from [Fork](https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch/)  


  "Attention Is All You Need"에서 제안된 모델에 Max/Avg Pooling 적용(default = max) 후 소프트맥스 함수를 거쳐 예측값을 출력.
