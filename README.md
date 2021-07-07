# Sentiment Analysis with Transformer

## Fork
https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch

## 수정사항
- [x] *.ipynb files -> *.py files  
- [x] numpy를 제외한 모든 작업을 gpu로 이관  
- [x] tqdm_notebook -> tqdm  
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
pip install -r requirements.txt
```
  
2. Download Dataset
```
# In Ubuntu
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
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