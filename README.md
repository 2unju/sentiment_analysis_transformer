# Sentiment Analysis with Transformer

## Fork
https://github.com/radoslawkrolikowski/sentiment-analysis-pytorch

## 수정사항
- [x] *.ipynb files -> *.py files  
- [x] 불필요한 데이터 삭제  
- [x] numpy를 제외한 모든 작업을 gpu로 이관  
- [ ] tqdm_notebook -> tqdm  
- [x] checkpoint 추가
## Experiments
### Dataset
IMDb 영화 리뷰 데이터셋 50,000개 이용  
Train : Validation : Test = 0.7 : 0.2 : 0.1  
<img src="https://user-images.githubusercontent.com/77797199/125013576-056b2400-e0a7-11eb-8561-8127a763f90e.png">  

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

<img src="https://user-images.githubusercontent.com/77797199/125012902-e4ee9a00-e0a5-11eb-9c85-194c1855cbb8.PNG">
<img src="https://user-images.githubusercontent.com/77797199/125012846-d2746080-e0a5-11eb-843d-bb2fa259dfa7.PNG">
  
|Train acc|Validation acc|Test acc|
|:---:|:---:|:---:|
|0.906|0.887|0.872|
