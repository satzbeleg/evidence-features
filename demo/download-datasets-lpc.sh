#!/bin/bash

# create folder for data and results
DATAFOLDER=./data
mkdir -p "${DATAFOLDER}"

# download files
# (80k) Hochdeutsch in DE, CH, AT, LI, BE, LU, NA, EU
# (80k) Dialekte: nds/nds-nl, gsw, bar, pfl, ltz, ksh, lim
declare -a FILES=(
    deu-de_web-public_2019_10K 
    deu-ch_web-public_2019_10K 
    deu-at_web-public_2019_10K 
    deu-li_web_2019_10K 
    deu-be_web_2013_10K 
    deu-lu_web_2019_10K 
    deu-na_web_2019_10K 
    deu-eu_web_2017_10K
    nds_wikipedia_2021_10K
    nds-nl_wikipedia_2021_10K
    gsw_wikipedia_2021_10K
    bar_wikipedia_2021_10K
    pfl_wikipedia_2021_10K
    ltz_wikipedia_2021_10K
    ksh_wikipedia_2021_10K
    lim_wikipedia_2021_10K
)

for FILENAME in ${FILES[*]} ; do
    echo "$FILENAME"
    wget -nc --no-check-certificate "https://pcai056.informatik.uni-leipzig.de/downloads/corpora/${FILENAME}.tar.gz"
    tar -xf "${FILENAME}.tar.gz"
    cat "${FILENAME}/${FILENAME}-sentences.txt" | cut -d$'\t' -f2- > "./${DATAFOLDER}/${FILENAME}.txt"
    rm -rf "${FILENAME}"
    rm -rf "${FILENAME}.tar.gz"
done
