#!/bin/bash
NOTEBOOKS=*.ipynb
for f in $NOTEBOOKS
do
  echo "Processing $f..."
  jupyter nbconvert --to custom --template=custom_notebook.tpl $f --TagRemovePreprocessor.remove_cell_tags='{"hidden"}'
  base="${f%.*}"
  mv $base.txt ../$f
done

# Move datasets up
DATASETS=data_*.txt
for f in $DATASETS
do
  mv $f ../$f
done