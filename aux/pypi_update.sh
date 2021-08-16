# Update to PyPI (remember to update setup.cfg!)
cd /Users/tdougan/Dropbox\ \(HMS\)/Research/General/waltlabtools 
rm -r dist src/waltlabtools.egg-info
python3 -m build
twine upload dist/*
sleep 5
pip install waltlabtools --upgrade
sleep 10
pip install waltlabtools --upgrade
conda list


# Sphinx-apidoc
cd /Users/tdougan/Dropbox\ \(HMS\)/Research/General/waltlabtools/docs
sphinx-apidoc -f --implicit-namespaces -o source ../src/waltlabtools
make clean
make html