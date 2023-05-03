#!/bin/bash

wget -nc "https://media.dwds.de/dta/download/dtak/2020-10-23/normalized/gesamt.zip"
unzip -n gesamt.zip
rm gesamt.zip
mv gesamt data
