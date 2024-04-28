#importing libraries
from audiocraft.models import MusicGen
import streamlit as st
import os 
import torch
import torchaudio
import base64

#loading the model and caching the model
@st.cache_resource
def load_model():
    model=MusicGen.get_pretrained("facebook/musicgen-small")
    return model
#generating the music
def generate_music(description, duration: int):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]

def save_audio(samples: torch.mps):
    sample_rate=32000
    save_path="/Users/surajsatheesh/MCA/Second Semester/project/output"
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def download(bin_file, file_label="File"):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

#ui
st.set_page_config(
    page_title="Text2Music"
)
def main():
    st.title("Text to Music")
    st.write("## 📝 How to use it? \n1. 🎹 In the `Enter Prompt` section, describe the type of music you wish to generate. \n2. 🎵 Click on `Generate Music` to let Text2Music work its magic. - 🎧 You can preview the generated music in the `Generated Music Preview` section.")

    text_area=st.text_area("Enter the prompt")
    time_slider=st.slider("Select time duration (in seconds)", 5, 120, 10)#min, max and default

    #showing the output of the description
    if text_area and time_slider:
        st.json(
            {
                "Description: ": text_area,
                "Time: ": time_slider
            }
        )
        st.subheader("Generated Music")

        music=generate_music(text_area,time_slider)

        print("Music: ",music)
        save_file=save_audio(music)
        save_music_file = save_audio(music)
        audio_filepath = 'output/audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(download(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ =="__main__":
    main()