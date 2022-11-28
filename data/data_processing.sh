mkdir cache
mkdir graph
cd ./script
wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_500d.pkl.bz2
bzip2 -d enwiki_20180420_500d.pkl.bz2

python step_0_data_conversion.py
python step_1_entity_extraction.py
python step_2_data_standardize.py 
python step_3_extract_srl.py 
python step_4_graph_construction.py 
python step_5_clean_data.py

cd ../
mv SEESAW* ./cache
mv *standard* ./cache