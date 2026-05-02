# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_pipe = pipeline("text-generation", 
                          model="pranavpsv/genre-story-generator-v2")
    story_text = story_pipe(scenario)
    story = story_results[0]['generated_text']
    st.write(f"**Story:** {story}")   
    return story_text

# text2audio
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)    
    return audio_data

# main part
def main():
    
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
        
main()

