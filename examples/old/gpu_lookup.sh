#!/bin/bash
#lspci | grep VGA | grep -i NVIDIA | awk '{print $1}' | xargs -i lspci -v -s {}
#PCI_BUS_DEVICE=`lspci | grep VGA | grep -i NVIDIA`

#greps VGA and NVIDIA on lspci to get the device line for the GPU.
#uses regex grep to pull only the gpu associated with the chip
#uses table https://en.wikipedia.org/wiki/CUDA#GPUs_supported
#to switch and output the correct cuda vserison
PCI_BUS_DEVICE=`lspci | grep VGA | grep -i NVIDIA | grep -Eo "G.[0-9]{2}."`

#https://en.wikipedia.org/wiki/CUDA#GPUs_supported
#!!!!!!!!!!!!1.0-1.1 are not supported by this script!!!!!!!!!
#1.0 G80
#1.1 G92, G94, G96, G98, G84, G86
#1.2 GT218, GT216, GT215
if echo $PCI_BUS_DEVICE | grep -q 'GT21'; then
        out=12;
fi
#1.3 GT200, GT200b
if echo $PCI_BUS_DEVICE | grep -q 'GT200'; then
        out=13;
fi
#2.0 GF100, GF110
if echo $PCI_BUS_DEVICE | grep -q 'GF100\|GF110'; then
        out=20;
fi
#2.1 GF104, GF106 GF108, GF114, GF116, GF117, GF119
if echo $PCI_BUS_DEVICE | grep -q 'GF104\|GF106 GF108\|GF114\|GF116\|GF117\|GF119'; then
        out=21;
fi
#3.0 GK104, GK106, GK107
if echo $PCI_BUS_DEVICE | grep -q 'GK104\|GK106\|GK107'; then
        out=30;
fi
#3.2 GK20A
if echo $PCI_BUS_DEVICE | grep -q 'GK20A'; then
        out=32;
fi
#3.5 GK110, GK208
if echo $PCI_BUS_DEVICE | grep -q 'GK110\|GK208'; then
        out=35;
fi
#3.7 GK210
if echo $PCI_BUS_DEVICE | grep -q 'GK210'; then
        out=37;
fi
#5.0 GM107, GM108 
if echo $PCI_BUS_DEVICE | grep -q 'GM107\|GM108'; then
        out=50;
fi
#5.2 GM200, GM204, GM206
if echo $PCI_BUS_DEVICE | grep -q 'GM200\|GM204\|GM206'; then
        out=52;
fi
#5.3 GM20B
if echo $PCI_BUS_DEVICE | grep -q 'GM20B'; then
        out=53;
fi
#6.0 GP100
if echo $PCI_BUS_DEVICE | grep -q 'GP100'; then
        out=60;
fi
#6.1 GP102, GP104, GP106, GP107, GP108
if echo $PCI_BUS_DEVICE | grep -q 'GP102\|GP104\|GP106\|GP107\|GP108'; then
        out=61;
fi
#6.2 GP10B
if echo $PCI_BUS_DEVICE | grep -q 'GP10B'; then
        out=62;
fi
#7.0 GV100
if echo $PCI_BUS_DEVICE | grep -q 'GV100'; then
        out=70;
fi
#7.1 GV10B
if echo $PCI_BUS_DEVICE | grep -q 'GV10B'; then
        out=71;
fi

echo $out;
