wget https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v2.1/json -O webnlg_dataset.zip
unzip webnlg_dataset.zip -d webnlg_dataset
mv webnlg_dataset/*/*/*/* webnlg_dataset
rm webnlg_dataset.zip
cd webnlg_dataset
rm -rf */
mkdir raw
mv *.json raw