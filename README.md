# Cog Wrapper for SeamlessExpressive

SeamlessExpressive is a multilingual speech translation model that preserves the original vocal styles and prosody. It retains the nuances of speech, ensuring translations maintain the speaker's unique expressions and intonation. 

See the original [paper](https://scontent-ist1-1.xx.fbcdn.net/v/t39.2365-6/406941874_247486308347770_2317832131512763077_n.pdf?_nc_cat=102&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=w7nmCz1SmMkAX9G6vjb&_nc_ht=scontent-ist1-1.xx&oh=00_AfCjk9EB4gXYQZgbun4dzcRj5FEFiSJKf2tStwkQogBIHQ&oe=65EB8F69) and [model page](https://huggingface.co/facebook/seamless-expressive) and [repository](https://github.com/facebookresearch/seamless_communication) for more details.

## How to use the API

You need to have Cog and Docker installed to run this model locally. To build the docker container with cog and run a prediction:

```
cog predict -i audio_input=@sample.mp3 -i target_lang="German" -i duration_factor=1.1
```

To start a server and send requests to your locally or remotely deployed API:
```
cog run -p 5000 python -m cog.server.http
```

To translate and synthesize audio using SeamlessExpressive, you need to provide an audio file and specify your translation preferences. The API input arguments are as follows:

- **audio_input:** Path to the input audio file that you want to translate and synthesize.  
- **source_lang:** The original language of your input audio. Supported languages are English, French, Spanish, German, Italian and Chinese (Mandarin). Default value is English.  
- **target_lang:** The target language for the output audio. Supported languages are English, French, Spanish, German, Italian and Chinese (Mandarin). Default value is French.  
- **duration_factor:** Adjusts the timing to better match the target language's speech rhythm. Recommendations include 1.0 for English, Chinese, Spanish; 1.1 for German and Italian; and 1.2 for French. Default is 1.0.  

## References
```
@inproceedings{seamless2023,
   title="Seamless: Multilingual Expressive and Streaming Speech Translation",
   author="{Seamless Communication}, Lo{\"i}c Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-juss{\`a}, Maha Elbayad, Hongyu Gong, Francisco Guzm{\'a}n, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, Mary Williamson",
  journal={ArXiv},
  year={2023}
}
```