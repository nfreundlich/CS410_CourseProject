#!/usr/bin/env bash
printf "Update package on pip.\n\n"
printf "WARNING:\nThis script will ERASE previous build/dist/egg, create a new version and upload it to pypi.\n"
read -p "Continue (y/n) [n]:" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    printf "Ok...\n"
else
    printf "Exiting script.\n\n"
    exit
fi

printf "\nListing existing distribution\n\n"
ls -ltra | grep dist
ls -ltra | grep feature_mining.egg-info
ls -ltra | grep build

printf "\nCleaning existing distribution\n\n"
rm -rf ./dist/*
rmdir ./dist/
rm -rf ./feature_mining.egg-info/*
rmdir ./feature_mining.egg-info/
rm -rf ./build/*
rmdir ./build/


read -p "Create new distribution (y/n) [n]:" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    printf "Ok...\n"
else
    printf "Exiting script.\n\n"
    exit
fi

printf "Create new distribution\n\n"
python setup.py sdist bdist_wheel

read -p "Upload to pypi (y/n) [n]:" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    printf "Ok...\n"
else
    printf "Exiting script.\n\n"
    exit
fi

printf "\nUpload to pypi\n\n"
python -m twine upload dist/* --skip-existing

printf "\n\nDone...\n\n"
