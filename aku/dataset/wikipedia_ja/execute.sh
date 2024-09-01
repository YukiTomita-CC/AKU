git clone https://github.com/ku-nlp/python-textformatting.git
cd python-textformatting
python setup.py install
cd ..

pip install -r requirements/requirements_dataset.txt

python aku/dataset/wikipedia_ja/wikipedia_ja.py
