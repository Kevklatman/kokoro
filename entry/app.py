import spaces
from kokoro import KModel, KPipeline
import gradio as gr
import os
import random
import torch
import numpy as np
from audio_effects import add_breathiness, add_tenseness, add_jitter, add_sultry, apply_emotion_effects

CUDA_AVAILABLE = torch.cuda.is_available()
models = {gpu: KModel().to('cuda' if gpu else 'cpu').eval() for gpu in [False] + ([True] if CUDA_AVAILABLE else [])}
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# Audio effects imported from audio_effects.py






@spaces.GPU(duration=30)
def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)

def generate_first(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE, 
                  breathiness=0.0, tenseness=0.0, jitter=0.0, sultry=0.0):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    
    # Collect all audio chunks and phoneme strings
    all_audio_chunks = []
    all_phonemes = []
    
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps)-1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
                
            # Apply emotion effects
            audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
                
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Retrying with CPU. To avoid this error, change Hardware to CPU.')
                audio = models[False](ps, ref_s, speed)
                # Apply emotion effects
                audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
            else:
                raise gr.Error(e)
        
        # Add this chunk to our collection
        all_audio_chunks.append(audio.numpy())
        all_phonemes.append(ps)
    
    if not all_audio_chunks:
        return None, ''
    
    # Combine all audio chunks into a single numpy array
    combined_audio = np.concatenate(all_audio_chunks)
    combined_phonemes = '\n'.join(all_phonemes)
    
    # Return the combined audio
    return (24000, combined_audio), combined_phonemes

# Arena API
def predict(text, voice='af_heart', speed=1, breathiness=0.0, tenseness=0.0, jitter=0.0, sultry=0.0):
    return generate_first(text, voice, speed, use_gpu=False, 
                         breathiness=breathiness, tenseness=tenseness, jitter=jitter, sultry=sultry)[0]

def tokenize_first(text, voice='af_heart'):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ''

def generate_all(text, voice='af_heart', speed=1, use_gpu=CUDA_AVAILABLE,
                breathiness=0.0, tenseness=0.0, jitter=0.0, sultry=0.0):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in pipeline(text, voice, speed):
        # Safer version that won't cause index errors
        pack_length = len(pack)
        ref_idx = min(len(ps)-1, pack_length-1)  # Base index, safely bounded

        ref_s1 = pack[ref_idx]
        # Only try to get other references if they exist
        #ref_s2 = pack[max(0, min(ref_idx-1, pack_length-1))]
        #ref_s3 = pack[max(0, min(ref_idx-2, pack_length-1))]
        #ref_s4 = pack[max(0, min(ref_idx-3, pack_length-1))]

        # Mix the references
        mixed_ref = ref_s1
        try:
            if use_gpu:
                audio = forward_gpu(ps, mixed_ref, speed)
            else:
                audio = models[False](ps, mixed_ref, speed)
                
            # Apply emotion effects
            audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
                
        except gr.exceptions.Error as e:
            if use_gpu:
                gr.Warning(str(e))
                gr.Info('Switching to CPU')
                audio = models[False](ps, mixed_ref, speed)
                # Apply emotion effects
                audio = apply_emotion_effects(audio, breathiness, tenseness, jitter, sultry)
            else:
                raise gr.Error(e)
        yield 24000, audio.numpy()
        if first:
            first = False
            yield 24000, torch.zeros(1).numpy()

with open('en.txt', 'r') as r:
    random_quotes = [line.strip() for line in r]

def get_random_quote():
    return random.choice(random_quotes)

def get_frankenstein():
    return "I am by birth a Genovese, and my family is one of the most distinguished of that republic. My ancestors had been for many years counsellors and syndics, and my father had filled several public situations with honour and reputation."

def get_gatsby():
    with open('gatsby5k.md', 'r') as r:
        return r.read().strip()

def apply_preset(preset_name):
    preset = VOICE_PRESETS.get(preset_name)
    if not preset:
        return None
    
    # Find the voice choice display name
    voice_value = preset['voice']
    voice_display = next((k for k, v in CHOICES.items() if v == voice_value), None)
    
    return (
        voice_display,
        preset['speed'],
        preset['breathiness'],
        preset['tenseness'],
        preset['jitter'],
        preset['sultry']
    )

CHOICES = {
'🇺🇸 🚺 Heart ❤️': 'af_heart',
'🇺🇸 🚺 Bella 🔥': 'af_bella',
'🇺🇸 🚺 Nicole 🎧': 'af_nicole',
'🇺🇸 🚺 Aoede': 'af_aoede',
'🇺🇸 🚺 Kore': 'af_kore',
'🇺🇸 🚺 Sarah': 'af_sarah',
'🇺🇸 🚺 Nova': 'af_nova',
'🇺🇸 🚺 Sky': 'af_sky',
'🇺🇸 🚺 Alloy': 'af_alloy',
'🇺🇸 🚺 Jessica': 'af_jessica',
'🇺🇸 🚺 River': 'af_river',
'🇺🇸 🚹 Michael': 'am_michael',
'🇺🇸 🚹 Fenrir': 'am_fenrir',
'🇺🇸 🚹 Nolan': 'am_nolan',
'🇺🇸 🚹 Kevin': 'am_kevin',
'🇺🇸 🚹 Josh': 'am_josh',
'🇺🇸 🚹 Adam': 'am_adam',
'🇺🇸 🚹 Jack': 'am_jack',
'🇺🇸 🚹 Phoenix': 'am_phoenix',
'🇬🇧 🚺 Ruby': 'bf_ruby',
'🇬🇧 🚺 Selene': 'bf_selene',
'🇬🇧 🚺 Silvia': 'bf_silvia',
'🇬🇧 🚹 Michael': 'bm_michael',
'🇬🇧 🚹 Kevin': 'bm_kevin',
'🇬🇧 🚹 Daniel': 'bm_daniel',
}

# Voice presets from README.md with predefined emotion settings
VOICE_PRESETS = {
    'literature': {
        'voice': 'af_bella',  # Bella
        'speed': 1.1,
        'breathiness': 0.1,
        'tenseness': 0.1,
        'jitter': 0.15,
        'sultry': 0.1
    },
    'articles': {
        'voice': 'af_sky',    # Sky
        'speed': 1.0,
        'breathiness': 0.15,
        'tenseness': 0.5,
        'jitter': 0.3,
        'sultry': 0.1
    }
}

for v in CHOICES.values():
    pipelines[v[0]].load_voice(v)

TOKEN_NOTE = '''
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
'''

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label='Output Audio', interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button('Generate', variant='primary')
    with gr.Accordion('Output Tokens', open=True):
        out_ps = gr.Textbox(interactive=False, show_label=False, info='Tokens used to generate the audio, up to 75,000 character context length (approximately 25 pages).')
        tokenize_btn = gr.Button('Tokenize', variant='secondary')
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button('Predict', variant='secondary', visible=False)

STREAM_NOTE = ['⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`.']
STREAM_NOTE = '\n\n'.join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label='Output Audio Stream', interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button('Stream', variant='primary')
        stop_btn = gr.Button('Stop', variant='stop')
    with gr.Accordion('Note', open=True):
        gr.Markdown(STREAM_NOTE)
        gr.DuplicateButton()

API_OPEN = True
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label='Input Text', info=f"Supports up to 25 pages of text (approximately 75,000 characters)", lines=10)
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice', info='Quality and availability vary by language')
                use_gpu = gr.Dropdown(
                    [('ZeroGPU 🚀', True), ('CPU 🐌', False)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    info='GPU is usually faster, but has a usage quota',
                    interactive=CUDA_AVAILABLE
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            
            with gr.Accordion('Emotion Controls', open=False):
                with gr.Row():
                    breathiness = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label='Breathiness')
                    tenseness = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label='Tenseness')
                with gr.Row():
                    jitter = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label='Jitter/Tremor')
                    sultry = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label='Sultry')
            random_btn = gr.Button('🎲 Random Quote 💬', variant='secondary')
            with gr.Row():
                gatsby_btn = gr.Button('🥂 Gatsby 📕', variant='secondary')
                frankenstein_btn = gr.Button('💀 Frankenstein 📗', variant='secondary')
            
            # Voice preset buttons
            with gr.Row():
                gr.Markdown("### Voice Presets")
            with gr.Row():
                literature_preset_btn = gr.Button('📚 Literature (Bella)', variant='secondary')
                articles_preset_btn = gr.Button('📰 Articles (Sky)', variant='secondary')
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ['Generate', 'Stream'])
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    
    # Preset button handlers
    literature_preset_btn.click(
        fn=lambda: apply_preset('literature'),
        inputs=[],
        outputs=[voice, speed, breathiness, tenseness, jitter, sultry]
    )
    articles_preset_btn.click(
        fn=lambda: apply_preset('articles'),
        inputs=[],
        outputs=[voice, speed, breathiness, tenseness, jitter, sultry]
    )
    generate_btn.click(fn=generate_first, inputs=[text, voice, speed, use_gpu, breathiness, tenseness, jitter, sultry], outputs=[out_audio, out_ps])
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
    stream_event = stream_btn.click(fn=generate_all, inputs=[text, voice, speed, use_gpu, breathiness, tenseness, jitter, sultry], outputs=[out_stream])
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(fn=predict, inputs=[text, voice, speed, breathiness, tenseness, jitter, sultry], outputs=[out_audio])

if __name__ == '__main__':
    # Configuration for running behind a reverse proxy with HTTPS
    # The app will only be accessible from the local server
    # A reverse proxy (like Nginx) should be configured to handle HTTPS and forward requests to this app
    app.queue(api_open=API_OPEN).launch(server_name="127.0.0.1", server_port=40001, show_api=API_OPEN)
