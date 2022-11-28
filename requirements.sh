conda create -n SEESAW python=3.8
conda activate SEESAW
pip install -U torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U transformers==4.11.3
pip install -U torch-geometric==2.0.3 torch-sparse==0.6.12 torch-scatter==2.0.9 torch-cluster==1.5.9 -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
pip install -U pip setuptools wheel
pip install -U spacy==3.2.2
pip install -U click==8.0.4
python -m spacy download en_core_web_sm==3.2.0
pip install -U allennlp==2.9.0
pip install -U allennlp-models==2.9.0
pip install -U wikipedia2vec==1.0.5
pip install -U nltk==3.6.5
pip install -U rdflib==6.1.1