#!/bin/bash

jupyter nbconvert --to notebook --inplace --execute answers/RegularizedRegression.ipynb
mv answers/*.csv .