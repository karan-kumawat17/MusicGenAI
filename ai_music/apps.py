import logging
from django.apps import AppConfig
from django.core.cache import cache
import pickle
from django.conf import settings
import os
import pandas as pd


# Get an instance of a logger
logger = logging.getLogger(__name__)

class YourAppConfig(AppConfig):
    name = 'ai_music'

    def ready(self):
        try:
            lyrics_path = os.path.join(settings.BASE_DIR, 'ai_music', 'data', 'sample_song_lyrics_set.obj')
            embeddings_path = os.path.join(settings.BASE_DIR, 'ai_music', 'data', 'embeddings_indices.obj')

            # Load the data and store it in the cache
            with open(lyrics_path, 'rb') as f:
                sample_artists_set, lyrics_set = pickle.load(f)
            with open(embeddings_path, 'rb') as f:
                embeddings, arr_song_idx, arr_lyrics_idx = pickle.load(f)

            # Store data in the cache
            cache.set('sample_artists_set', sample_artists_set, None)
            cache.set('lyrics_set', lyrics_set, None)
            cache.set('embeddings', embeddings, None)
            cache.set('arr_song_idx', arr_song_idx, None)
            cache.set('arr_lyrics_idx', arr_lyrics_idx, None)
            logger.info('Stored all data in cache successfully')

        except Exception as e:
            # If any exception occurs, log it as an error
            logger.error('Failed to load and cache data', exc_info=True)
