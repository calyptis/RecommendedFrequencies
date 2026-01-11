import unittest

from recommended_frequencies.spotify.library import get_spotipy_instance


class Spotify(unittest.TestCase):
    def test_get_client(self):
        sp = get_spotipy_instance()
        response = sp.track("46q5BtHso0ECuTKeq70ZhW")
        response = response if response != {} else None
        self.assertIsNotNone(response)
