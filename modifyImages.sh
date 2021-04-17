#!/bin/bash

ls "heads/input/" | while read image
do
    imgWithoutExtention=$(echo "$image" | rev | cut -d'.' -f 2- | rev)
    # controllo esistenza del pt
    echo "testo immagine: $imgWithoutExtention"
    if test -z $(ls heads/target/* | egrep "heads/target/$imgWithoutExtention.pt")
    then
        echo "Rimuovo heads/target/$imgWithoutExtention"
        rm "heads/input/$image"
    fi
done
