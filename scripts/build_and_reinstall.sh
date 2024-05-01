rm -rf ./dist
python setup.py bdist_wheel
rm -rf *.egg-info
pip uninstall ms2vec -y && pip install dist/*.whl