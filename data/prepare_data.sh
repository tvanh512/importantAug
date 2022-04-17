# Download Google Speech Command data
wget -P ./data/ http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p ./data/SpeechCommands/speech_commands_v0.02
tar -xf ./data/speech_commands_v0.02.tar.gz -C ./data/SpeechCommands/speech_commands_v0.02

# Download GSC-Musan data
wget -P ./data/ https://zenodo.org/record/6066174/files/SpeechCommands_Musan.tar.gz?download=1
mv ./data/'SpeechCommands_Musan.tar.gz?download=1' ./data/SpeechCommands_Musan.tar.gz
mkdir -p ./data/SpeechCommands_Musan
tar -xf ./data/SpeechCommands_Musan.tar.gz -C ./data/SpeechCommands_Musan

#Create GSC-QUT test set
wget -P ./data/ https://data.researchdatafinder.qut.edu.au/dataset/a0eed5af-abd8-441b-b14a-8e064bc3d732/resource/35cd737a-e6ad-4173-9aee-a1768e864532/download/qutnoisehome.zip
mkdir -p /data/QUT-NOISE/
unzip ./data/qutnoisehome.zip -d ./data/QUT-NOISE/
python ./data/create_GSC_QUT_testset.py