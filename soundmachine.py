import vlc #type:ignore
import time

def player(path):
    instance = vlc.Instance('--no-video', '--quiet')

    media = instance.media_new(path)
    player = instance.media_player_new()
    player.set_media(media)


    player.play()
    time.sleep(3)
    player.stop()
