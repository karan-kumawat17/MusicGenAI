from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from .models import ImageSubmission
from .forms import ImageUploadForm
from PIL import Image
import scipy
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, MusicgenForConditionalGeneration
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import re
import pickle
import math
import numpy as np
import pandas as pd
import nltk
#first time usage: download addtional packages form nltk first:
#nltk.download()
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .spotify_api.main import get_token


def index(request):
    return render(request, 'index.html')


def image_to_music(request):
    # Initialize the form instance
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image to the database
            image_submission = form.save()

            # Load the image for processing
            fs = FileSystemStorage()
            filename = fs.save(image_submission.image.name, image_submission.image)
            raw_image = Image.open(fs.path(filename)).convert('RGB')

            # Process the image to generate music
            img_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            img_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            inputs = img_processor(raw_image, return_tensors="pt")
            out = img_model.generate(**inputs)
            txt = img_processor.decode(out[0], skip_special_tokens=True)

            # Generate music from the caption
            audio_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            audio_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
            inputs = audio_processor(text=[txt], padding=True, return_tensors="pt")
            audio_values = audio_model.generate(**inputs, max_new_tokens=2000)
            sampling_rate = audio_model.config.audio_encoder.sampling_rate

            # Write the generated music to a file
            music_filename = 'music_output.wav'
            music_path = fs.path(music_filename)
            music_file_url = fs.url(music_filename)
            print("Music file URL:", music_file_url)
            scipy.io.wavfile.write(music_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
            image_url = fs.url(filename)

            # Provide the path for the generated music to the template
            context = {'music_file_url': music_file_url, 'image_url': image_url}
            return render(request, 'image_to_music_result.html', context)
    else:
        form = ImageUploadForm()

    return render(request, 'image_to_music.html', {'form': form})

#############################################################################


def image_to_text(path_to_image):
  img_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
  img_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

  raw_image = Image.open(path_to_image).convert('RGB')

  inputs = img_processor(raw_image, return_tensors="pt")

  out = img_model.generate(**inputs)
  print(img_processor.decode(out[0], skip_special_tokens=True))
  txt = img_processor.decode(out[0], skip_special_tokens=True)

  return txt  


# Get closest lyrics lines matches from user text input
def text_get_similar_lyrics_lines(user_text_input, embeddings, arr_lyrics_idx, model_name = "all-distilroberta-v1"):
    model = SentenceTransformer(model_name)
    input_emb = model.encode(user_text_input, convert_to_tensor=True)
    res_cos_sim = util.semantic_search(input_emb, embeddings, score_function=util.cos_sim, top_k=100)
    # Convert results and mapped lyrics id as pd dataframe
    res_df = pd.DataFrame(res_cos_sim[0])
    res_df.rename(columns = {'corpus_id':'lyrics_id'}, inplace = True)
    res_df['lyrics_line'] = arr_lyrics_idx[res_df['lyrics_id']]
    return res_df


def lyrics_id_mapping(res_df, arr_lyrics_idx):
    arr_lyrics_id = res_df['lyrics_id'].to_numpy()
    arr_idx = arr_lyrics_id.astype(int)
    arr_song_row_idx = arr_lyrics_idx[arr_idx]
    res_df['song_idx'] = arr_song_row_idx
    return res_df


# Suppress utterances which have low similarity scores
def score_low_sim_weighting(df, threshold = 0.9, weight_low_sim = 1):
    df['score_weighted'] = df['score'].apply(lambda x: x * weight_low_sim if x < threshold else x)
    return df


# Re-rank on songs level based on average lyrics line scores
def songs_ranking(df_results_lyrics_mapped):
    res = df_results_lyrics_mapped.groupby('song_idx')['score_weighted'].mean()
    res = res.sort_values(ascending=False)
    return res


# Combine songs information to ranked songs
def combine_songs_info(s_songs_ranking, sample_artists_set, results_limit = 10):
    df_songs_candidates = sample_artists_set.filter(items = s_songs_ranking.index, axis=0)
    df_songs_candidates['score'] = s_songs_ranking
    df_songs_candidates['song_idx'] = s_songs_ranking
    res_df = df_songs_candidates[['artist', 'title', 'score']][:10]
    return res_df


# Overall function to generate songs ranking based on lyrics lines semantic textual similarity 
def similar_songs_ranked(user_input, embeddings, sample_artists_set, lyrics_set, arr_song_idx):
    df_results_lyrics = text_get_similar_lyrics_lines(user_input, embeddings, lyrics_set)
    df_results_lyrics_mapped = lyrics_id_mapping(df_results_lyrics, arr_song_idx)
    df_results_lyrics_mapped = score_low_sim_weighting(df_results_lyrics_mapped)
    s_songs_ranking = songs_ranking(df_results_lyrics_mapped)
    df_results_songs = combine_songs_info(s_songs_ranking, sample_artists_set)
    return df_results_songs, df_results_lyrics_mapped
    
    
# Helper function to support getting songs/ lyrics results

# Look up relevant lyrics lines an their similarity scores
def lyrics_scores_lookup(song_id, df_results_lyrics_mapped):
    res = df_results_lyrics_mapped[df_results_lyrics_mapped['song_idx'] == song_id][['lyrics_line', 'score']]
    res = res.sort_values(by=['score'], ascending=False)
    return res


# Generate output on both songs and lyrics level, as a list of dictionaries
def similar_songs_lyrics_ranked(df_results_songs, df_results_lyrics_mapped):

    result_list = []

    for song_id in df_results_songs.index:
        song_title = df_results_songs['title'].loc[song_id]
        song_artist = df_results_songs['artist'].loc[song_id]
        song_score = df_results_songs['score'].loc[song_id]
        song_id = song_id
        df_lyrics_scores = lyrics_scores_lookup(song_id, df_results_lyrics_mapped)
        d_lyrics = dict(zip(df_lyrics_scores['lyrics_line'], df_lyrics_scores['score']))
        dict_object = {"song_id": song_id, "artist":song_artist, "song_title":song_title, "song_score":song_score, "lyrics_scores":d_lyrics}
        result_list.append(dict_object)
    
    return result_list

##########################################################################################


# Your recommend_music function modified to fit in the Django views.py structure
def recommend_music(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            image_path = fs.path(filename)

            # Retrieve your data from cache
            sample_artists_set = cache.get('sample_artists_set')
            embeddings = cache.get('embeddings')
            arr_lyrics_idx = cache.get('arr_lyrics_idx')
            arr_song_idx = cache.get('arr_song_idx')

            # Check if data is not None, otherwise, it means it's not loaded
            if sample_artists_set is None or sample_artists_set.empty:
                raise ValueError("Data is not loaded properly.")

            # Here you would use your image_to_text function
            user_input = image_to_text(image_path)

            # Now find similar songs based on the image-derived text
            df_results_songs, df_results_lyrics_mapped = similar_songs_ranked(
                user_input, embeddings, sample_artists_set, arr_lyrics_idx, arr_song_idx)
            result = similar_songs_lyrics_ranked(df_results_songs, df_results_lyrics_mapped)

            # Convert the result to a DataFrame to render it in your template
            results_df = pd.DataFrame(result)
            token = get_token()
            results_list = results_df.to_dict('records')
            print(results_list)

            context = {'results': results_list, 'token': token}
            # Finally, render a template with the results
            return render(request, 'recommend_music_result.html', context)
    else:
        form = ImageUploadForm()
    return render(request, 'recommend_music.html', {'form': form})
