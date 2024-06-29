IMAGE TO MUSIC RECOMMENDATION : 

HOW TO RUN : 


After unzipping the folder


1. Create and activate a virtual environment by running the following command. First install virtualenv. 


        pip install virtualenv


        Then to create a virtual environment, run:


        python -m venv myenv
        Then activate the virtual environment. On windows run:
        
        .\myenv\Scripts\activate
        
        On macOS and Linux, run:


        source myenv/bin/activate


        
2. Navigate to your project directory in the command line where your Python files are located and run:


        pip install -r requirements.txt




This will install all the necessary dependencies.


3. Ensure the database is setup well by running the following command in the project directory:


        python manage.py migrate

4. Download the data files, either for hindi or english dataset from https://drive.google.com/drive/folders/1vyJirorAjWouXbNoI8hyxlTsd5jMZfmq?usp=sharing

and paste the .obj files in the data folder(you should paste 2 files, either for hindi or english).


5. Now, start the server:


        python manage.py runserver

Dataset Information : 


Dataset is scraped song lyrics from Genius API (Miller, 2020) and song features from Spotify API (Lamere, 2022). The song list is based on the Wasabi dataset (Michael Fell, 2020), a large song corpus with metadata. We cannot directly use the Wasabi dataset, as the lyrics cannot be saved for download due to copyright issues. The available data are all trained embedding results. With a large corpus of over 2 million songs, we filtered out only songs with Spotify ID and from artists in the top 100 billboards since 1958. With the time limitation and the rate limit from APIs, we only work with 100,000 songs for our corpus of study.


Dataset Cleaning : 


With the data scraped from the Spotify API based on the list of songs from Wasabi Songs dataset, approximately 2 million songs were filtered by the popularity of the artist from billboard's top 100 since 1948. We look at the attributes of the song including danceability, energy, speechiness, acousticness, liveness, and valence. The distribution of these properties shows quite a good distribution of the data and the songs that we have for our model. Although speechiness and liveness tend to be on the lower side, it does not really impact our songs distribution of mood range that much as the liveness parameter is the detection of the presence of the audience in the audio, and speechiness represents the spoken words that are not melodic in the audio. These 2 do not seem to impact the range of mood that much versus danceability, energy, acousticness, and valence which distributes quite well in the middle. Hence, it can be safe to assume that we can get a normally distributed songs' attribute if we sample from the dataset for our model representation.


Information Retrieval Model : 


* Feature Engineering & Embedding Generation : With our lyrics dataset, we first employed a pre-trained RoBERTa model (Yinhan Liu, 2019) through the Sentence Transformer framework to generate fixed-sized dense vectors. From the lyrics embeddings, we can then compare the transformed text query using the same model to find the embeddings’ similarity. We will also fine-tune our model using the song lyrics and annotations from the Genius community as an attempt to generate embeddings that could have higher accuracy rates in terms of semantic similarity. Once we get the embedding of the lyrics from the sample songs that we sampled from the dataset, we then performed analysis on what is the best number of clusters using elbow plot. We then perform the clustering on our dataset using MiniBatchKmean to help with the speed of the clustering considering the embedding size and samples.


* Semantic Similarity : With all the embeddings generated from the models, a K-nearest neighbor model (KNN) is fit to the matrix, when a new embedding is generated from a user query, we could retrieve the top n candidates from the KNN model. We planned to use the widely-adopted cosine distance between the embeddings as the metric for semantic textual similarity (Briggs, 2021).




* Results Ranking : We will incorporate compositional similarity scores in the ranked results, including overall song similarity score using average similarity from lyrics lines-user query pairs, as well as the scores of individual lines. To make the average similarity score more sensitive to sentences that are highly similar to a user query, we have made a function to penalize lines with low similarity scores.










Resources : 


*  Lamere, P. (2022, Jun 26). Spotipy. From Read the Docs:  Spotipy.


*  Markelle Kelly, K. M. (2021, Feb). An Exploration of BERT for Song Classification and. From kaimalloy: BERT.


*  Michael Fell, E. C. (2020, Mar 15). Cornell University. From Arxiv: Arxiv.


*  Miller, J. W. (2020). lyrics genius. From Read the Docs: lyricsgenius.


*  Reimers, N. (2022). Sbert.net. From Sentence-Transformer: Sentence Transformer.


*  Yinhan Liu, M. O. (2019, Jul 26). RoBERTa: A Robustly Optimized BERT Pretraining Approach. From Arxiv: Robust BERT.


*  Briggs, J. (2021). NLP similarity metrics | towards data science. Similarity Metrics in NLP, from Similarity Metrics.












IMAGE TO MUSIC GENERATION(MusicGenAI) :


Dataset Information :


* Training Dataset : We use 20K hours of licensed music to train our model. Specifically, we rely on an internal dataset of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data collections2 with respectively 25K and 365K instrument-only music tracks. All datasets consist of full-length music sampled at 32 kHz with metadata composed of a textual description and information such as the genre, BPM, and tags. We downmix the audio to mono unless stated otherwise.


* Evaluation Dataset :  For the main results and comparison with prior work, we evaluate the proposed method on the MusicCaps benchmark [Agostinelli et al., 2023]. MusicCaps is composed of 5.5K samples (ten-second long) prepared by expert musicians and a 1K subset balanced across genres. We report objective metrics on the unbalanced set, while we sample examples from the genre-balanced set for qualitative evaluations. For melody evaluation and the ablation studies, we use samples from an in-domain held out evaluation set of 528 music tracks, with no artist overlap with the training set.




Information Retrieval Model : 


This whole task was done in two phases. First is Image-to-Text Generation & then taking valuable information from the text and converting it into music.


For the first task, we used BLIP model. BLIP: Bootstrapping Language-Image Pre-training.


BLIP, a new VLP(Vision Language Pre-Training) framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones.


  







Model Architecture : 


We employ a visual transformer (Dosovitskiy et al., 2021) as our image encoder, which divides an input image into patches and encodes them as a sequence of embeddings, with an additional [CLS] token to represent the global image feature.


In order to pre-train a unified model with both understanding and generation capabilities, we propose multimodal mixture of encoder-decoder (MED), a multi-task model which can operate in one of the three functionalities:


1. Unimodal encoder


2. Image-grounded text encoder


3. Image-grounded text decoder




Image-Text Retrieval : 


We evaluate BLIP for both image-to-text retrieval (TR) and text-to-image retrieval (IR) on COCO and Flickr30K (Plummer et al., 2015) datasets. We finetune the pre-trained model using ITC and ITM losses. To enable faster inference speed, we follow Li et al. (2021a) and first select k candidates based on the image-text feature similarity, and then rerank the selected candidates based on their pairwise ITM scores.


Now comes the part of text-to-audio generation:


Our model consists in an autoregressive transformer-based decoder[Vaswanietal.,2017], conditioned on a text or melody representation.The (language) model is over the quantized units from anEnCodec [Défossezetal.,2022]audio tokenizer,which provides high fidelity reconstruction from a low frame rate discrete representation. Compression models such as [Défossezetal.,2022,  Zeghidouretal.,2021] employ Residual Vector Quantization (RVQ) which results in several parallel streams.




Model Conditioning : 


* Text Conditioning : Given a textual description matching the input audio X, we compute a conditioning tensor C ∈ RTC×D with D being the inner dimension used in the autoregressive model. Generally, there are three main approaches for representing text for conditional audio generation. Kreuk et al. [2022] proposed using a pre trained text encoder, specifically T5 [Raffel et al., 2020]. Chung et al. [2022] show that using instruct-based language models provide superior performance. Lastly, Agostinelli et al. [2023], Liu et al. [2023], Huang et al. [2023a], Sheffer and Adi [2023] claimed that joint text-audio representation, such as CLAP [Wu* et al., 2023], provides better-quality generations.




* Melody Conditioning :  While text is the prominent approach in conditional generative models nowadays, a more natural approach for music is conditioning on a melodic structure from another audio track or even whistling or humming. Such an approach also allows for an iterative refinement of the model’s output. To support that, we experiment with controlling the melodic structure via jointly conditioning on the input’s chromagram and text description.






Model & Hyperparameters : 


* Audio Tokenization Model : We use a non-causal five layers EnCodec model for 32 kHz monophonic audio with a stride of 640, resulting in a frame rate of 50 Hz, and an initial hidden size of 64, doubling at each of the model’s five layers. The embeddings are quantized with a RVQ with four quantizers, each with a codebook size of 2048.




* Transformers Model : We train autoregressive transformer models at different sizes: 300M, 1.5B, 3.3B parameters. We use a memory efficient Flash attention [Dao et al., 2022] from the xFormers package [Lefaudeux et al., 2022] to improve both speed and memory usage with long sequences. We use the 300M-parameter model for all of our ablations. We train on 30-second audio crops sampled at random from the full track. We train the models for 1M steps with the AdamW optimizer [Loshchilov and Hutter, 2017], a batch size of 192 examples, β1 = 0.9, β2 = 0.95, a decoupled weight decay of 0.1 and gradient clipping of 1.0. Wefurther rely on D-Adaptation based automatic step-sizes [Defazio and Mishchenko, 2023] for the 300M model as it improves model convergence but showed no gain for the bigger models. We use a cosine learning rate schedule with a warmup of 4000 steps. Additionally, we use an exponential moving average with a decay of 0.99. 




* Text Pre-Processing : Kreuk et al. [2022] proposed a text normalization scheme, in which stop words are omitted and the remaining text is lemmatized. We denote this method by text-normalization. When considering musical datasets, additional annotation tags such as musical key, tempo, type of instruments, etc. are often available.




* Codebook Patterns & Conditioning : We use the “delay” interleaving pattern. This translates 30 seconds of audio into 1500 autoregressive steps. For text conditioning, we use the T5 [Raffel et al., 2020] text encoder, optionally with the addition of the melody conditioning.