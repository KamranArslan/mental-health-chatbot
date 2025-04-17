from collections import Counter

class EmotionFusion:
    def __init__(self):
        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
    
    def fuse_emotions(self, text_emotion=None, speech_emotion=None, face_emotion=None):
        # Collect all available emotions (excluding None values)
        emotions = []
        if text_emotion is not None:
            emotions.append(text_emotion)
        if speech_emotion is not None:
            emotions.append(speech_emotion)
        if face_emotion is not None:
            emotions.append(face_emotion)
            
        if not emotions:
            return "neutral"  # Default emotion if no inputs
            
        # If only one emotion is available, return it directly
        if len(emotions) == 1:
            return emotions[0]
            
        # Use majority voting for multiple emotions
        emotion_counts = Counter(emotions)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        return dominant_emotion 