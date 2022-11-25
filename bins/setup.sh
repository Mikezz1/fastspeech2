#download LjSpeech

mkdir ./data
mkdir ./checkpoints
# wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
# tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
# mv LJSpeech-1.1 ./data/LJSpeech-1.1
# rm LJSpeech-1.1.tar.bz2

gdown https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx
mv train.txt ./data/

#download Waveglow
gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mv waveglow_256channels_ljs_v2.pt ..vocoder/waveglow/pretrained_model/waveglow_256channels.pt

# load pre-computed melspecs, pitch and energy vectors
gdown https://drive.google.com/u/0/uc?id=1lwes4q0G_KSw9V4kKf7YNDTJMDnwSw3t
tar -xvf melspecs.tar.gz
mv melspecs ./data

gdown https://drive.google.com/u/0/uc?id=1GxcW3Zjp9gWeTEbaZ0mb-QfObSqHzkZ4
tar -xvf pitch.tar.gz
mv pitch ./data



gdown https://drive.google.com/u/0/uc?id=1pgeQcBnDNZlND3lLXlIJ1bdrOaN_T7ZL
tar -xvf energy.tar.gz
mv energy ./data

#download alignments
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip 
mv alignments ./data/alignments/
rm alignments.zip 
