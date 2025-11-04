import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline
from pyannote.audio import Pipeline


def audio_process(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(16000)
    audio.export('processed_audio.wav', format='wav')
    return 'processed_audio.wav'


def diarization_audio(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token="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")
    diarization = pipeline({'audio': audio_file})
    return diarization


def transcribe_audio(audio_file, start_time, end_time):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as file:
        audio_data = recognizer.record(file, offset=start_time, duration=end_time - start_time)
        try:
            text = recognizer.recognize_google(audio_data)
        except:
            text = "Nothing is transcribed"
    return text


def sentiment_analysis(text):
    s_pipeline = pipeline("sentiment-analysis")
    sentiments = s_pipeline(text)
    return sentiments


def sentiment_analysis_audio(audio_file):
    processed_audio = audio_process(audio_file)
    diarization = diarization_audio(processed_audio)
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time, end_time = turn.start, turn.end
        transcription = transcribe_audio(processed_audio, start_time, end_time)
        if transcription is not None:
            sent_analysis = sentiment_analysis(transcription)
            results.append({
                "start time": start_time,
                "end time": end_time,
                "speaker": speaker,
                "transcription": transcription,
                "sentiment": sent_analysis
            })
    return results


audio_file = "test_audio.wav"
sent_results = sentiment_analysis_audio(audio_file)
for res in sent_results:
    print(f"Time:{res['start time']} - {res['end time']}")
    print(f"Speaker: {res['speaker']}")
    print(f"Transcription: {res['transcription']}")
    print(f"Sentiment: {res['sentiment']}")
    print("--------------------------------")